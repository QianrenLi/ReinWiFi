import os
import json
import sys
import threading
import numpy as np
import copy

abs_path = os.path.dirname(os.path.abspath(__file__))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph

FRACTION = "fraction"

class envCap:
    def __init__(self, graph:Graph, env_path: str, cost_func = None) -> None:
        self.graph = graph
        self.action_keys = ["cw", "aifs", "txop", "fraction"]
        self.action_key_zipper = {"cw": 0, "aifs": 0, "txop": 0, "fraction": 0}
        self.action_format = {"cw-aifs-txop-fraction":{}}
        self.state_format = {"rtt":{}}
        self.env_graph = {}
        self.cost_func = cost_func
        self.env_path = env_path
        self.scenario = "scenario_1"

        pass

    def active_stream_num(self):
        num = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        num += 1
        return str(num)

    def zip_action_value(self, action_key_zipper):
        return "%d-%d-%d-%d" % (action_key_zipper["cw"], action_key_zipper["aifs"], action_key_zipper["txop"], action_key_zipper["fraction"])

    ## Formulate the environment graph
    def system_formation(self):
        if self.scenario not in self.env_graph:
            self.env_graph.update({self.scenario:{"controls":{}, "system_return":{}}})
        return self


    ## Collect data to environment graph
    def collect_data(self, controls, system_return, index):
        self.system_formation()
        # self.graph.show()

        self.env_graph[self.scenario]["controls"].update({str(index): controls.copy()})
        self.env_graph[self.scenario]["system_return"].update({str(index): system_return.copy()})
        return self


    def load_data(self):
        try:
            with open(self.env_path, "r") as f:
                self.env_graph = json.load(f)
        except IOError:
            print("File not accessible")
            with open(self.env_path, "w+") as f:
                pass

        return self


    ## Save data to file
    def save_data(self):
        with open(self.env_path, "w") as f:
            json.dump(self.env_graph, f, indent=4)
        return self

    def get_state(self,active_stream_num:int, stream_name:str, action:str):
        pass
        # return self

    # Load data from file
    def action2ind(self, controls, action_space, graph:Graph):
        """
        From a given action, return the index of the action in the action space
        """
        index = 0
        fraction = controls[FRACTION]

        his_num = len(action_space[0])
        cw_aifs_num = [len(action_space[1]),  len(action_space[2])]
        fraction_idx = np.where(np.array(action_space[0]) == fraction)[0][0]
        index += fraction_idx

        # cw_idx = np.where(np.array(action_space[1]) == cw)[0][0]
        # aifs_idx = np.where(np.array(action_space[2]) == aifs)[0][0]

        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    prot, sender, receiver = link_name.split("_")
                    if "file" not in stream["file_name"]:
                        port, tos = stream_name.split("@")
                        control_name = prot + "_" + tos + "_" + sender + "_" + receiver
                        for _idx, action_name in enumerate(["cw", "aifs"]):
                            _val = controls[control_name][action_name]
                            _ind = np.where(np.array(action_space[ _idx + 1 ]) == _val)[0][0]
                            index += _ind * his_num
                            his_num *= cw_aifs_num[_idx]
        index = max(index, 1)
        return index

    def ind2state(self, ind):
        state = copy.deepcopy(self.env_graph[self.scenario]["system_return"][str(ind)])
        for stream_name in state:
            for port_name in state[stream_name]:
                if "rtt" in state[stream_name][port_name]:
                    if state[stream_name][port_name]["rtt"] > 0:
                        noise_rtt = state[stream_name][port_name]["rtt"] + np.random.normal(0, 0.0005)
                        # print(noise_rtt)
                        state[stream_name][port_name].update({"rtt": noise_rtt})
        # TODO: add random noise
        return state