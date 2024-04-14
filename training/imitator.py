## Construct a imitator model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random

from training.environment import envCap
from training.netUtil import envNet

abs_path = os.path.dirname(os.path.abspath(__file__))
current = abs_path
parent = os.path.dirname(current)
sys.path.append(parent)

from transmission_graph import Graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class imitator:
    def __init__( self,  env : envCap, graph: Graph, action_num, state_num, data_type = "dict" ) -> None:
        self.graph = graph
        self.env = env

        self.action_num = action_num
        self.state_num = state_num

        self._rtt_threshold = 0.04

        self.imitator_net = envNet( action_num, 512, state_num ).to(device)
        self.imitator_optimizer = optim.Adam( self.imitator_net.parameters(), lr = 0.001 )
        self.imitator_criteria = nn.MSELoss().to(device)
        # self.imitator_criteria = nn.HuberLoss().to(device)

        self.data_type = data_type


    def train( self, controls, system_returns ):
        action = []
        state = []
        for control, system_return in zip(controls, system_returns):
            # print(control)
            # print(system_return)
            action.append(self.action_interpreter( self.graph, control, self.action_num ))
            state.append(self.state_interpreter( self.graph, system_return, self.state_num ))

        action_tensor = torch.from_numpy( np.array( action ) ).float().to(device)
        state_tensor = torch.from_numpy( np.array( state ) ).float().to(device)

        predict_state = self.imitator_net( action_tensor )
        # print(state_tensor)
        # print(predict_state)
        self.imitator_optimizer.zero_grad()
        loss = self.imitator_criteria( predict_state, state_tensor )
        for param in self.imitator_net.parameters():
            if param.grad is not None and param.grad.nelement > 0:
                torch.nn.utils.clip_grad_value_([param], 100)
        loss.backward()
        self.imitator_optimizer.step()

        return loss.item()

    def predict( self, controls ):
        action = self.action_interpreter( self.graph, controls, self.action_num )

        action_tensor = torch.from_numpy( np.array( [ action ] ) ).float().to(device)
        predict_state = self.imitator_net( action_tensor )
        return predict_state.detach().cpu().numpy()

    def eval( self, controls, system_return ):
        with torch.no_grad():
            predict_state = self.predict( controls )
        state = self.state_interpreter( self.graph, system_return, self.state_num )
        # print(predict_state[0])
        # print(state)
        # loss = np.mean( np.power( np.log10(predict_state[0] / state), 2) ) 
        loss = np.mean( np.power( predict_state[0] - state, 2) ) 
        # distance = np.abs(predict_state[0] - state)
        # loss = np.sum(distance >= 0.5)
        return loss

    def store_params( self, path ):
        torch.save( self.imitator_net.state_dict(), path )

    def load_params( self, path ):
        self.imitator_net.load_state_dict( torch.load( path ) )

    def action_interpreter(self, graph:Graph, i_controls :dict, action_num, action_keys = ["cw", "aifs"], is_sorted = False):
        if self.data_type == "dict":
            FRACTION = "fraction"
            controls = i_controls.copy()
            fraction = controls[FRACTION]
            controls.pop(FRACTION)
            actions = np.zeros(action_num)
            actions[0] = fraction

            _idx = 1
            for device_name, links in graph.graph.items():
                for link_name, streams in links.items():
                    prot, sender, receiver = link_name.split("_")
                    is_rtt_active = False
                    for stream_name, stream in streams.items():
                        port, tos = stream_name.split("@")
                        control_entry = prot + "_" + tos + "_" + sender + "_" + receiver
                        if control_entry in controls:
                            for key in action_keys:
                                actions[_idx] = controls[control_entry][key]
                                _idx += 1
                            controls.pop(control_entry)
                            is_rtt_active = True

                    if not is_rtt_active and not is_sorted:
                        for key in action_keys:
                            _idx += 1
            assert( controls == {} )
            return actions
        elif self.data_type == "list":
            return i_controls
        else:
            raise( "Error imitator data type" ) # type: ignore

    def state_interpreter(self, graph:Graph, system_return ,state_num, is_sorted = False):
        """
        Only output rtt
        """
        if self.data_type == "dict":
            states = np.zeros(state_num)
            _idx = 0
            for device_name, links in graph.graph.items():
                for link_name, streams in links.items():
                    is_rtt_active = False
                    for stream_name, stream in streams.items():
                        if "file" not in stream["file_name"]:
                            is_rtt_active = True
                            states[_idx] = system_return[link_name][stream_name]["rtt"] * 1000 if system_return[link_name][stream_name]["rtt"]  < 0.04 else 40
                            _idx += 1
                    if not is_rtt_active and not is_sorted:
                        _idx += 1
            return states
        elif self.data_type == "list":
            return system_return
        else:
            raise( "Error imitator data type" ) # type: ignore

    @staticmethod
    def state_deinterpreter( graph, states, system_return, is_sorted = False ):
        """
        Only output rtt
        """
        _idx = 0
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                is_rtt_active = False
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        is_rtt_active = True
                        system_return[link_name][stream_name]["rtt"] = states[_idx] / 1000
                        _idx += 1
                if not is_rtt_active and not is_sorted:
                    _idx += 1
        return system_return