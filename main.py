import test_case as tc
import threading
import os
import time
import argparse
import numpy as np

from training.txAgent import tx
from training.txController import txController
# from training.system2train import wlanRLController
from training.system2trainV3 import wlanRLController
from training.environment import envCap
from training.imitator import imitator
from tap import Connector


abs_path = os.path.dirname(os.path.abspath(__file__))


is_transmission_stop = threading.Event()
is_ctr_dominate = True
experiment_name = "test"


def start_ctl_thread(wireless_controller: txController, wireless_tx: tx, args: argparse , ctl_type = "train", point_val = 0.6):

    tx_thread = threading.Thread(target=wireless_tx.transmission_thread)
    if ctl_type == "train":
        ctl_thread = threading.Thread(target=wireless_controller.control_thread, args=(1000,))
        # wireless_controller.net_controller.test_mode = True
        wireless_controller.net_controller.UCB = False
        train_thread = threading.Thread(target=wireless_controller.training_thread, args = ("%s%s%s.pt" % (abs_path, args.ctlModelFolder , args.model),  ))
        train_thread.start()
    elif ctl_type == "test":
        wireless_controller.net_controller.test_mode = True
        ctl_thread = threading.Thread(target=wireless_controller.control_thread, args=(200,))
    else:
        print("Wrong control type")
        return

    tx_thread.start()
    ctl_thread.start()


    if is_ctr_dominate:
        while True:
            is_transmission_stop.wait()
            if wireless_controller.is_ctl_stop.is_set():
                break
            time.sleep(1)
            ## Set control continuously
            wireless_tx.init_thottle = wireless_controller.heuristic_fraction * 600
            wireless_tx._set_manifest()
            tx_thread = threading.Thread(target=wireless_tx.transmission_thread)
            tx_thread.start()
    else:
        is_transmission_stop.wait()
        wireless_controller.is_ctl_stop.wait()

def gen_ctl_list(control):
    control_list = []
    fraction_val = 0
    for key in control:
        if key == "fraction":
            fraction_val = control["fraction"]
        else:
            control_list.append(control[key]["cw"])
            control_list.append(control[key]["aifs"])
    control_list.insert(0, fraction_val)
    return control_list

def interference_identifier(wireless_controller: txController, wireless_tx: tx,action_num:int,env_classifier:imitator, ctl_prot = "wlx"):
    his_duration = wireless_tx.per_transmission_duration
    wireless_tx.per_transmission_duration = 10

    # controls = wireless_controller._generate_controls([0.6] + [7,2] * action_num, c_type="list")
    controls = wireless_controller._generate_controls([0.3] + [7,2] * action_num, c_type="list")
    print(controls)
    edca_params = wireless_controller._edca_default_params(controls)
    wireless_controller._set_edca_parameter(edca_params)
    fraction = controls["fraction"]
    wireless_controller.heuristic_fraction = fraction
    wireless_tx.init_thottle = fraction * 600

    while True:
        try:
            wireless_tx.prepare_transmission(ctl_prot)
            break
        except:
            continue

    tx_thread = threading.Thread(target=wireless_tx.transmission_thread)
    tx_thread.start()
    is_transmission_stop.wait()

    action_rtt = gen_ctl_list(controls) + wireless_tx.rtt_values
    print(action_rtt)
    if_index = env_classifier.predict(action_rtt)[0]
    print("interference index" , np.argmax(if_index))
    wireless_tx.per_transmission_duration = his_duration
    return np.argmax(if_index)


def load_network(wlanController: wlanRLController, args):
    try:
        wlanController.load_params("%s%s%s.pt" % (abs_path, args.ctlModelFolder , args.model))
    except FileNotFoundError:
        print("%s%s%s.pt" % (abs_path, args.ctlModelFolder , args.model))
        print("not exist such model")
        exit()
    except Exception as e:
        print(e)
        exit()


def get_action_num_from_graph( graph ):
    action_num = 0
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            for stream_name, stream in streams.items():
                if "file" not in stream["file_name"]:
                    action_num += 1
    return action_num

from tools.fileRx import receiver
def start_file_recv_thread( file_name ):
    host = "192.168.3.82"
    port = 15555
    ## start thread
    threading.Thread(target=receiver, args=(host, port, file_name,)).start()

def sync_rtt_files( graph ):
    isSend = False 
    conn = Connector()
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            if not isSend:
                prot, sender, receiver = link_name.split("_")
                conn.batch(sender, "send_file", {"target_ip": args.controllerIp, "file_name": "../stream-replay/logs/rtt-*.txt"})
                isSend = True
                print(sender)
    conn.executor.wait(0.5).apply()
    print("Send file to controller")

def set_experiment_scenario(idx, wireless_tx:tx, links:list):
    def add_file(wireless_tx: tx, link_name: str, port: int):
        wireless_tx.graph.ADD_STREAM(
            link_name,
            port_number=port,
            file_name="file_75MB.npy",
            duration=[0, wireless_tx.per_transmission_duration],
            thru=0,
            tos=96,
            name="File",
        )

    def add_proj(wireless_tx: tx, link_name: str, port: int, target_rtt=18):
        wireless_tx.graph.ADD_STREAM(
            link_name,
            port_number=port,
            file_name="proj_6.25MB.npy",
            duration=[0, wireless_tx.per_transmission_duration],
            thru=7 * 8,
            tos=128,
            target_rtt=target_rtt,
            name="Proj-%s" % link_name,
        )

    def add_cast(wireless_tx: tx, link_name: str, port: int, target_rtt = 24):
        wireless_tx.graph.ADD_STREAM(
            link_name,
            port_number=port,
            # file_name="cast_1.5625MB.npy",
            file_name="cast_3.125MB.npy",
            duration=[0, wireless_tx.per_transmission_duration],
            thru=3.5 * 8,
            tos=128,
            target_rtt=target_rtt,
            name="Proj-%s" % link_name,
        )

    def add_interference(wireless_tx: tx, link_name:str, port:int, thru = 50):
        wireless_tx.graph.ADD_STREAM(
            link_name,
            port_number = port,
            file_name ="file_75MB.npy",
            duration = [0, wireless_tx.per_transmission_duration],
            thru = thru,
            tos = 128,
            name ="Interference",
        )

    if idx == 0:
        wireless_tx.ignore_idx = -1
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)

    elif idx == 1:
        add_interference(wireless_tx, links[0], 6203)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)

    elif idx == 2:
        add_interference(wireless_tx, links[0], 6203, 100)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)

    elif idx == 3:
        wireless_tx.ignore_idx = -1
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        # add_cast(wireless_tx, lists[1], 6203)
        add_cast(wireless_tx, links[0], 6202)   

    elif idx == 4:
        add_interference(wireless_tx, links[0], 6203)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        # add_cast(wireless_tx, lists[1], 6204)
        add_cast(wireless_tx, links[3], 6202)

    elif idx == 5:
        add_interference(wireless_tx, links[0], 6203, 100)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        # add_cast(wireless_tx, lists[1], 6204)
        add_cast(wireless_tx, links[3], 6202)

    elif idx == 6:
        wireless_tx.ignore_idx = -1
        add_cast(wireless_tx, links[1], 6203)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        add_cast(wireless_tx, links[0], 6202)   

    elif idx == 7:
        add_interference(wireless_tx, links[0], 6203)
        add_cast(wireless_tx, links[1], 6204)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        add_cast(wireless_tx, links[3], 6202)

    elif idx == 8:
        add_interference(wireless_tx, links[0], 6203, 100)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        add_cast(wireless_tx, links[1], 6204)
        add_cast(wireless_tx, links[3], 6202)

    elif idx == 9:
        add_interference(wireless_tx, links[0], 6203, 150)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        add_cast(wireless_tx, links[1], 6204)
        add_cast(wireless_tx, links[3], 6202)

    elif idx == 10:
        add_interference(wireless_tx, links[0], 6203, 25)
        add_proj(wireless_tx, links[2], 6201)
        add_file(wireless_tx, links[2], 6200)
        # add_cast(wireless_tx, lists[1], 6203)
        add_cast(wireless_tx, links[3], 6202)   
    
def exec_cw_ctl(
        wireless_controller: txController, 
        wireless_tx: tx, 
        wlanController: wlanRLController, 
        args: argparse, 
        ctl_prot = "wlx", 
        ):
    if args.base:
        start_ctl_thread(wireless_controller, wireless_tx, args , "line_search")
        return
    elif args.throttle:
        wireless_tx.init_thottle = args.val * 600
        wireless_tx.prepare_transmission(ctl_prot)
        start_ctl_thread(wireless_controller, wireless_tx, args, "fix_point", point_val = args.val)
    elif args.linesearch:
        start_ctl_thread(wireless_controller, wireless_tx, args, "line_search")
    elif args.heuristic:
        start_ctl_thread(wireless_controller, wireless_tx, args, "heuristic")
    else:
        load_network(wlanController, args)
        wireless_controller.net_controller = wlanController
        wireless_controller.net_controller.cluster_idx = 0
        if args.test:
            start_ctl_thread(wireless_controller, wireless_tx, args, "test")
        elif args.explore:
            start_ctl_thread(wireless_controller, wireless_tx, args, "explore")
        else:
            start_ctl_thread(wireless_controller, wireless_tx, args, "train")

def main(args):
    ## Receiver RTT thread
    folder = f'./tools/logs/{args.rttLogFolder}'
    start_file_recv_thread(folder) # start the receiver thread
    ## Setup graph
    graph, links = tc.cw_training_case_without_intereference()

    action_space = [
        [i / 20 for i in range(1, 20, 1)],
        [1, 3, 7, 15, 31, 63],  # CW value
        [2],                    # AIFSN
    ]

    rl_scheduler = wlanRLController(
        action_space[0],
        action_space[1],  # CW value
        action_space[2],  # AIFSN
        10000,
        graph,
        batch_size= 32 ,
        is_CDQN = False,
        is_k_cost= 0,
        gamma = 0.9,
        log_level = 1
    )

    ## Setup tx object
    wireless_tx = tx(graph, is_transmission_stop)
    wireless_tx.per_transmission_duration = 60
    wireless_tx.links = links
    wireless_tx.ignore_idx = 0
    _idx = args.scenario
    set_experiment_scenario(_idx, wireless_tx, links)

    ## Setup sockets
    ctl_prot = "wlx"
    wireless_tx.prepare_transmission(ctl_prot)

    ## Setup txController object
    wireless_controller = txController(graph, is_transmission_stop, wireless_tx.socks, action_space)
    wireless_controller.ctl_sig_interval_seconds = 2
    wireless_controller.is_control_dominate = is_ctr_dominate
    wireless_controller.communication_graph.display()
    wireless_controller.net_controller.cluster_idx = _idx
    
    exec_cw_ctl(wireless_controller, wireless_tx, rl_scheduler, args)

    sync_rtt_files(graph)

    print("Send file to controller")

if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgumentParser(description="RL controller for wireless network")
    parser.add_argument("--config", is_config_file=True, default="config.txt")
    parser.add_argument("--model", type=str, default="7-24-3")
    parser.add_argument("-v", "--val", type= float, default= 0.7)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--throttle", action="store_true")
    parser.add_argument("--linesearch", action="store_true")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("-s", "--scenario", type=int, default= 0)
    parser.add_argument("--ctlModelFolder", type=str, default="/training/model/")   
    parser.add_argument("--rttLogFolder", type=str, default="test", help="folder after './tools/logs/'")
    parser.add_argument("--controllerIp", type=str, default="192.168.3.82")
    args = parser.parse_args()    
    main(args)
