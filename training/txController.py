import os
import json
import sys
import ctypes
import numpy as np
import threading
import time
import queue


abs_path = os.path.dirname(os.path.abspath(__file__))

current = abs_path
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph
from tap import Connector
from tap_pipe import ipc_socket
from training.environment import envCap


## ====================================================================== ##


class txController:
    """
    This class is used to control the transmission of the network

    Func:
        Communicate with ongoing tx users (tx object from txAgent)
    """

    def __init__(
        self,
        graph: Graph,
        transmission_stop_flag: threading.Event,
        socks: list,
        action_space: list,
        control_interval = 1,
        train_int_max = 10,
        control_max = 50,
    ) -> None:
        """
        Initial controller with graph, which provide the communication necessary information, is_stop event which used to stop the controlling,
        and ipc_sockets used to communicate with ongoing device
        Args:
            graph (Graph): graph object
            is_stop (threading.Event): stop event
            socks (list): list of ipc_socket
            control_interval (int, optional): control interval. Defaults to 1 (s).
        """
        self.communication_graph = graph
        self.transmission_stop_flag = transmission_stop_flag
        self.socks = socks
        self.action_space = action_space
        self.ctl_sig_interval_seconds = control_interval
        self.maximum_control_times = control_max
        self.train_int_max = train_int_max

        self.is_line_search_policy = False  # With enable the Line control algorithm,
        # the throttle will compute according to the line search algorithm
        self.is_local_test = False
        self.CONTROL_ON = True
        self.is_control_dominate = False

        self.tos_to_ac = {"196": 0, "128": 1, "96": 2, "32": 3}
        self.throttle = {}
        self.system_return = {}
        self.heuristic_fraction = 0
        self.his_file_num = 0
        self.env_cap = None

        self.is_ctl_stop = threading.Event()
        self.is_graph_update = threading.Event()
        self.net_available = threading.Event()
        self.train_queue = queue.Queue(maxsize=100)
        self.is_send_thro = threading.Event()

        self.rtt_maximum = 1

        self.env_buffer = None
        self.net_controller = None


    def _loop_tx(self, sock: ipc_socket, *args):
        """
        Continuous transmitting to the remote ipc socket until the transmission is successful
        """
        _retry_idx = 0
        print("Collect\t", sock.link_name)
        while True:
            try:
                _buffer = sock.ipc_communicate(*args)
                break
            except Exception as e:
                print(e)
                if self.transmission_stop_flag.is_set():
                    _buffer = None
                    break
                _retry_idx += 1
                print("timeout\t", sock.link_name)
                continue
        return _buffer, _retry_idx
    
    ## Control component
    def _update_file_stream_nums(self):
        """
        Inbuilt function to calculate active file stream in graph
        """
        file_stream_nums = 0
        for device_name, links in self.communication_graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        "file" in stream["file_name"]
                        and self.communication_graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        file_stream_nums += 1
        return file_stream_nums

    def _throttle_calc(self):
        """
        Compute exact throttle (Throughput) of each file transmission
        """
        # detect whether the num of file stream changes
        reset_flag = False
        this_throttle_fraction = self.heuristic_fraction

        if this_throttle_fraction:
            port_throttle = self.communication_graph.update_throttle(
                this_throttle_fraction, reset_flag
            )
        else:
            port_throttle = None
        return port_throttle

    ## EDCA injection component
    def _reset_edca_params(self):
        params = {}
        ## iterate all stream in graph
        for device_name, links in self.communication_graph.graph.items():
            if "ind" in self.communication_graph.info_graph[device_name]:
                params.update({device_name : {
                    "ind": self.communication_graph.info_graph[device_name]["ind"]
                }
                })
        conn = Connector()
        for device_name in params.keys():
            conn.batch(device_name, "reset_edca", params[device_name])
        conn.executor.wait(0.1).apply()


    def _edca_default_params(self, controls: dict):
        """
        Setup edca params prepared to be injected to remote
        """
        params = {}

        for link_name_tos in controls.keys():
            if "_" in link_name_tos:
                prot, tos, tx_device_name, rx_device_name = link_name_tos.split("_")
                if tx_device_name in self.communication_graph.graph:
                    params[link_name_tos] = {
                        "ac": self.tos_to_ac[tos],
                        "cw_min": int(controls[link_name_tos]["cw"]),
                        "cw_max": int(controls[link_name_tos]["cw"]),
                        "aifs": int(controls[link_name_tos]["aifs"]),
                        "ind": self.communication_graph.info_graph[tx_device_name]["ind"],
                    }

        return params

    def _set_edca_parameter(self, params):
        """
        Inject EDCA parameter change to remote
        """
        conn = Connector()
        for link_name_tos in params:
            device_name = link_name_tos.split("_")[2]
            conn.batch(device_name, "modify_edca", params[link_name_tos])
            # print(params[link_name_tos])
            conn.executor.wait(0.1)
        conn.executor.wait(0.1).apply()
        return conn

    def _compute_maximum_index(self):
        fraction_num = len(self.action_space[0])
        cw_aifs_num = [len(self.action_space[1]), len(self.action_space[2])]
        self.maximum_control_times = 1
        for device_name, links in self.communication_graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        for _idx, action_name in enumerate(["cw", "aifs"]):
                            self.maximum_control_times *= cw_aifs_num[_idx]
                    else:
                        self.maximum_control_times *= fraction_num
        self.maximum_control_times += 1
        return self

    def _generate_controls(self, index: int, c_type = "collect"):
        """
        Generate controls for each iteration (Collect data only)
        1. EDCA Parameter
        2. Throttle
        """
        if c_type == "collect":
            control = {}
            fraction = 0
            # print("Control index", index)

            fraction_num = len(self.action_space[0])
            cw_aifs_num = [len(self.action_space[1]), len(self.action_space[2])]

            fraction_counter = index % fraction_num
            index = int(index / fraction_num)

            for device_name, links in self.communication_graph.graph.items():
                for link_name, streams in links.items():
                    prot, sender, receiver = link_name.split("_")
                    for stream_name, stream in streams.items():
                        port, tos = stream_name.split("@")
                        if "file" not in stream["file_name"]:
                            _control = {}
                            for _idx, action_name in enumerate(["cw", "aifs"]):
                                _action_val = self.action_space[_idx + 1][
                                    index % cw_aifs_num[_idx]
                                ]
                                _control.update({action_name: _action_val})
                                ## update action to graph
                                self.communication_graph.info_graph[device_name][link_name][
                                    stream_name
                                ].update({action_name: _action_val})
                                index = int(index / cw_aifs_num[_idx])
                            control.update(
                                {prot + "_" + tos + "_" + sender + "_" + receiver: _control}
                            )
                        else:
                            _action_val = self.action_space[0][fraction_counter]
                            self.communication_graph.info_graph.update({"fraction": _action_val})
                            control.update({"fraction": _action_val})
        elif c_type == "list":
            action_list = index ##format: [0.35  7.    3.    7.    9.    1.    3.   63.    3.]
            idx = 1
            control = {}
            fraction = 0
            for device_name, links in self.communication_graph.graph.items():
                for link_name, streams in links.items():
                    prot, sender, receiver = link_name.split("_")
                    for stream_name, stream in streams.items():
                        port, tos = stream_name.split("@")
                        if "file" not in stream["file_name"]:
                            _control = {}
                            for _idx, action_name in enumerate(["cw", "aifs"]):
                                _control.update({action_name: int(action_list[idx])})
                                idx += 1
                            control.update(
                                {prot + "_" + tos + "_" + sender + "_" + receiver: _control}
                            )
                        else:
                            control.update({"fraction": action_list[0]})
            assert(len(action_list) == idx)
        else:
            control = {}
            fraction_num = len(self.action_space[0])
            cw_aifs_num = [len(self.action_space[1]), len(self.action_space[2])]
            fraction_counter = index % fraction_num            
            _action_val = self.action_space[0][fraction_counter]
            control.update({"fraction": _action_val})
        return control

    def _validate_rtt(self, system_return):
        data_valid = True
        for inf in system_return:
            for _port in system_return[inf]:
                if system_return[inf][_port]["rtt"] is None:
                    data_valid = False
        return data_valid

    def training_thread(self, path: str):
        assert self.net_controller is not None
        loss_cum = 50
        loss = []
        while True:
            if self.net_controller.memory_counter > 32:
                # print("ready to train")
                _ = self.train_queue.get()
                # print("Get train queue")s
                if self.is_ctl_stop.is_set():
                    break
                if self.net_controller.train_counter % 5000 == 0:
                    self.net_controller.store_params(path)
                    self.net_controller.store_memory("./training/data/scenario11/%s.npy" % time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        ))
                self.net_available.wait()
                self.net_available.clear()
                # print("Start training")
                loss.append( self.net_controller.training_network() )
                if len(loss) > loss_cum:
                    print("Loss:\t", sum(loss) / len(loss))
                    loss.clear()
                self.net_available.set()

            else:
                print("Memory size:\t", self.net_controller.memory_counter )
                time.sleep(1)
    
    def _compute_cost(self):
        """
        Compute cost for each iteration
        """
        cost = 10
        for device_name, links in self.communication_graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    target_rtt = self.communication_graph.info_graph[device_name][link_name][
                        stream_name
                    ]["target_rtt"]
                    if (
                        self.communication_graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        if target_rtt != 0 and stream["rtt"] is not None:
                            if stream["rtt"] * 1000 - target_rtt > 0:
                                _cost = 10
                            else:
                                _cost = 0
                            cost += _cost
        cost -= self.heuristic_fraction * 10
        # print("cost", cost)
        return cost

    def control_thread(self , maximum_control_times = 10):  # create a control_thread
        """
        Control thread:

        The control thread is responsible for collecting data and controlling the system,
        default the control is implemented by _generate_control function.
        It is recommended to be reconstructed by the user to implement the control algorithm.
        """
        # start control and collect data
        system_return = {}
        throttle = {}
        controls = None
        state = None
        state_ = None

        control_times = 0
        is_restart = False
        last_control_time = 0
        cost_list = []
        cost_dict = {}
        self.maximum_control_times = maximum_control_times
        conn = Connector()
        self._reset_edca_params()
        self.is_ctl_stop.clear()
        self.net_available.set()
        if self.env_buffer is not None:
            self.env_buffer.load_data()
        
        ## Wait until tx start
        while self.transmission_stop_flag.is_set():
            time.sleep(0.1)
            continue
        ## 
        while True:

            control_times += 1
            ## collect data
            print("send statistics")
            for sock in self.socks:

                _buffer, _retry_idx = self._loop_tx(sock, "statistics")
                if _buffer is None:
                    break

                link_return = json.loads(str(_buffer.decode()))

                sys.stdout.flush()
                system_return.update({sock.link_name: link_return["body"]})
                time.sleep(0.1)

            if not self._validate_rtt(system_return):
                control_times -= 1
                continue

            ## Determine break -- Generally speaking, it is not a good choice to use thread unsafe parameter as a condition
            if not self.is_control_dominate:
                # print("Here")
                if self.transmission_stop_flag.is_set():
                    break
            else:
                if control_times > self.maximum_control_times:
                    break
                else:
                    if _buffer is None:
                        print("Here")
                        is_restart = True
                        # control_times = last_control_time - 1
                        continue
            print(system_return)
            ## update graph: activate and deactivate function
            self.communication_graph.update_graph(system_return)
            self.is_graph_update.set()

            if controls is not None and self.env_buffer is not None:
                if not is_restart:
                    last_control_time = control_times
                    print(system_return)
                    self.env_buffer.collect_data(
                        controls, system_return, control_times - 1
                    )

            ## Get edca parameter
            if self.net_controller is None:
                controls = self._generate_controls(control_times % 10, c_type = "heuristic")
            else:
                ## Network controller
                print("wait net available")
                self.net_available.wait()
                print("Net is available")
                self.net_available.clear()
                if state is not None:
                    state_ = self.net_controller.get_state()
                else:
                    self.net_controller.init_action_guess()
                    state = self.net_controller.get_state()
                if state_ is not None:
                    # cost = self.net_controller.get_cost(self.heuristic_fraction)
                    cost = self._compute_cost()
                    cost_list.append(cost)
                    print("COST: \t", cost)
                    if self.heuristic_fraction in cost_dict:
                        cost_dict[self.heuristic_fraction].append(cost)
                    else:
                        cost_dict.update({self.heuristic_fraction: [cost]})
                    self.net_controller.store_transition(
                        state, action_idx, cost, state_
                    )
                    state = state_
                controls, action_idx,original_action = self.net_controller.action_to_control(state)
                print("Generate Control")
                his_fraction = controls["fraction"]
                for _ in range(self.train_int_max):
                    if not self.train_queue.full():
                        self.train_queue.put(1)
                    else:
                        break
                # controls = self.net_controller.action_to_control()
                print("Finish Queue in")
                self.net_available.set()

            is_restart = False
            print(controls)
            self.heuristic_fraction = controls["fraction"]

            if self.CONTROL_ON:
                edca_params = self._edca_default_params(controls)
                ## Set edca parameter
                self._set_edca_parameter( edca_params)
                if port_throttle := self._throttle_calc():
                    # print(port_throttle)
                    throttle.update(port_throttle)
                    for sock in self.socks:
                        if sock.link_name in throttle.keys():
                            sock.ipc_transmit("throttle", throttle[sock.link_name])
                        else:
                            sock.ipc_transmit("throttle", {})
                else:
                    for sock in self.socks:
                        sock.ipc_transmit("throttle", {})
                    print("=" * 50)
                    print("Control Stop")
                    print("=" * 50)



            time.sleep(self.ctl_sig_interval_seconds)

        if self.env_buffer is not None:
            self.env_buffer.save_data()
        min_cost = 100
        outage_probability = 0
        print(cost_dict)
        for fraction_val in cost_dict:
            cost = np.mean(cost_dict[fraction_val])
            if min_cost > cost:
                min_cost = cost
                outage_probability = np.sum(np.array(cost_list) < 10) / len(cost_list)
        print("Minimum cost:\t",min_cost)
        print("outage_probability:\t", outage_probability)
        ## close sockets
        [sock.close() for sock in self.socks]
        self.socks.clear()
        self.is_ctl_stop.set()
        print("average_cost", np.mean(cost_list[100:])) # 100 is the warm up time
        print("control_times", control_times - 1)

    def random_control(self):  # create a control_thread
        """
        Control thread:

        The control thread is responsible for collecting data and controlling the system,
        default the control is implemented by _generate_control function.
        It is recommended to be reconstructed by the user to implement the control algorithm.
        """
        # start control and collect data
        system_return = {}
        throttle = {}
        controls = None
        state = None
        state_ = None

        control_times = 0
        his_control_times = 0
        is_restart = False
        update_counter = 0
        control_points = 4500
        DEFAULT_CONTROL = self._generate_controls(1, c_type = "collect")
        # edca_params = self._edca_default_params(DEFAULT_CONTROL)
        conn = Connector()
        # self._set_edca_parameter(conn, edca_params)
        # print(DEFAULT_CONTROL)
        # self.maximum_control_times = 20
        self._compute_maximum_index()
        
        # self._reset_edca_params()
        self.is_ctl_stop.clear()

        if self.env_buffer is not None:
            self.env_buffer.load_data()

        ## Wait until tx start
        while self.transmission_stop_flag.is_set():
            time.sleep(0.1)
            continue
        time.sleep(5)
        _system_return = {}
        # last_control_time = 0
        is_control_update = True
        while True:
            if is_control_update:
                his_control_times = control_times
                control_times = 1000 + update_counter
            ## collect data
            print("send statistics")
            for sock in self.socks:
                _buffer, _retry_idx = self._loop_tx(sock, "statistics")
                if _buffer is None:
                    break

                link_return = json.loads(str(_buffer.decode()))

                sys.stdout.flush()
                system_return.update({sock.link_name: link_return["body"]})
                time.sleep(0.1)

            if not self._validate_rtt(system_return):
                print("DATA not validate")
                is_control_update = False
                continue
            
            if not self.is_control_dominate:
                # print("Here")
                if self.transmission_stop_flag.is_set():
                    break
            else:
                if update_counter > control_points:
                    break
                else:
                    if _buffer is None:
                        print("Here")
                        is_restart = True
                        is_control_update = False
                        continue

            ## update graph: activate and deactivate function
            self.communication_graph.update_graph(system_return)
            self.is_graph_update.set()

            if controls is not None and self.env_buffer is not None:
                if not is_restart:
                    # last_control_time = control_times
                    is_control_update = True
                    print("Update Counter", update_counter)
                    print("Control Index ", his_control_times)
                    print(controls)
                    print(system_return)
                    self.env_buffer.collect_data(
                        controls, system_return, his_control_times
                    )
                    update_counter += 1
            
            print("----------Control Duration----------")
            ## Get edca parameter
            if self.net_controller is None:
                controls = self._generate_controls(control_times, c_type = "collect")
                print(controls)
                # controls = self._generate_controls(control_times, c_type = "heuristic")

            is_restart = False
            edca_params = self._edca_default_params(controls)
            # ## Set edca parameter
            self._set_edca_parameter(edca_params)
            self.heuristic_fraction = controls["fraction"]
            # self.heuristic_fraction = 0.95
            if self.CONTROL_ON:
                if port_throttle := self._throttle_calc():
                    # print(port_throttle)
                    throttle.update(port_throttle)
                    for sock in self.socks:
                        if sock.link_name in throttle.keys():
                            print("control throttle")
                            sock.ipc_transmit("throttle", throttle[sock.link_name])
                        else:
                            sock.ipc_transmit("throttle", {})
                else:
                    for sock in self.socks:
                        sock.ipc_transmit("throttle", {})
                    print("=" * 50)
                    print("Control Stop")
                    print("=" * 50)


            print("Control sent")
            time.sleep(self.ctl_sig_interval_seconds)
            
        if self.env_buffer is not None:
            self.env_buffer.save_data()

        ## close sockets
        [sock.close() for sock in self.socks]
        self.socks.clear()
        self.is_ctl_stop.set()

        print("Number of update", update_counter - 1)

    def send_fraction(self):
        throttle = {}
        self.is_send_thro.clear()
        while True:
            self.is_send_thro.wait()
            if port_throttle := self._throttle_calc():
                # print(port_throttle)
                throttle.update(port_throttle)
                for sock in self.socks:
                    if sock.link_name in throttle.keys():
                        print("control throttle")
                        sock.ipc_transmit("throttle", throttle[sock.link_name])
                    else:
                        sock.ipc_transmit("throttle", {})
            else:
                for sock in self.socks:
                    sock.ipc_transmit("throttle", {})
                print("=" * 50)
                print("Control Stop")
                print("=" * 50)

            self.is_send_thro.clear()

