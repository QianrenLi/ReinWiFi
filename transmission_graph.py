#!/usr/bin/env python3
import json

class Graph:
    """
    Structure used during start transmission and control.

    Args:
        graph (dict): containing information for transmission
        info_graph (dict): containing information for control
    """
    def __init__(self):
        """
        init Graph
        """
        self.graph = {}              
        self.info_graph = {}

    def ADD_DEVICE(self, device_name:str) -> None:                  
        """
        Add device

        Args:
            device_name (str): e.g "PC"
        """
        self.graph.update({device_name: {}})            
        self.info_graph.update({device_name: {}})
        pass

    def REMOVE_DEVICE(self, device_name:str) -> None:               
        """
        Remove device

        Args:
            device_name (str): e.g "PC"
        """
        del self.graph[device_name]
        del self.info_graph[device_name]

    def ADD_LINK(self, device_name:str, target_name:str, protocol:str, MCS:float) -> str:  # -> link name: e.g wlan_PC_phone
        """
        Add link according to tuple <source device name, target device name, transmission protocol> 

        Args:
            device_name (str): name of device which should be added before. e.g "PC","phone"
            target_name (str): name of target device which should be added by ADD_DEVICE before. e.g "PC", "phone"
            protocol (str): the transmission protocol along this link which is used to get ip address. e.g "wlan", "p2p"
            MCS (float): the MCS value of the transmission link

        Returns:
            str: the link name
        """
        link_name = protocol+'_'+device_name+'_'+target_name
        self.graph[device_name].update({link_name: {}})
        self.info_graph[device_name].update({link_name: {'MCS': MCS}})
        return link_name

    def REMOVE_LINK(self, link_name) -> None:                   # Remove link according to link name
        """
        Remove a existing link from graph

        Args:
            link_name (str): link name to be removed
        """
        device_name = link_name.split('_')[1]
        del self.graph[device_name][link_name]
        del self.info_graph[device_name][link_name]

    def ADD_STREAM(self, link_name: str, port_number:int, file_name:str, thru:float,
                   duration:list, tos=32, target_rtt=0, name = '') -> None:
        """
        Add stream to corresponding link 

        Args:
            link_name (str): name of link where the stream transmit on
            port_number (int): port number of stream, here the tx and rx use same port number
            file_name (str): name of file (.npy) which defines the transmission period and data size
            thru (float): the estimated throughput of the stream; When the stream is file, thru = 0 
            duration (list): [start time, end time] which defines when the stream start and stop
            tos (int, optional): the tos value which mapping the to EDCA parameter, e.g 32 -> AC3, 64 -> AC2, 96, 128 -> AC1, 192 -> AC0. Defaults to 32.
            target_rtt (int, optional): the QoS required rtt value. Defaults to 0.
            name (str, optional): name of stream, e.g "Speaker A". Defaults to ''.
        """
        device_name = link_name.split('_')[1]       # from link name to device name
        if type(port_number) == list:               # Add stream with multiple ports
            for _port_number in port_number:
                _name = name if name != '' else str(_port_number)+'@'+str(tos)
                self.graph[device_name][link_name].update({str(_port_number)+'@'+str(tos): {
                                                          'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})                 
                self.info_graph[device_name][link_name].update(
                    {str(_port_number)+'@'+str(tos): {"target_rtt": target_rtt, 'name': _name, "active": True}})
        else:
            _name = name if name != '' else str(port_number)+'@'+str(tos)
            self.graph[device_name][link_name].update({str(port_number)+'@'+str(tos): {
                                                      'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})
            self.info_graph[device_name][link_name].update(
                {str(port_number)+'@'+str(tos): {"target_rtt": target_rtt, 'name': _name, "active": True}})
        pass

    def REMOVE_STREAM(self, link_name:str, port_number: int, tos=132) -> None:
        """
        Remove stream based on link name and port number

        Args:
            link_name (str): name of link where stream transmitted
            port_number (int): the port of stream
            tos (int, optional): tos value of this stream. Defaults to 132.
        """
        device_name = link_name.split('_')[1]
        del self.graph[device_name][link_name][str(port_number)+'@'+str(tos)]
        del self.info_graph[device_name][link_name][str(
            port_number)+'@'+str(tos)]

    def UPDATE_DURATION(self, link_name:str , stream_name: str, duration: list) -> None:
        """
        Modify the duration value of stream

        Args:
            link_name (str): name of link where stream transmitted
            stream_name (str): name of stream
            duration (list): [Start point, End point]
        """
        device_name = link_name.split('_')[1]
        self.graph[device_name][link_name][stream_name]['duration'] = duration

    def associate_ip(self, device_name:str, protocol:str, ip_addr:str) -> None: 
        """
        Write ip addr of device over different protocol to info_graph

        Args:
            device_name (str): name of device, e.g "PC"
            protocol (str): name of protocol, e.g "wlan"
            ip_addr (str): ipv4 address, e.g "0.0.0.0"
        """
        self.info_graph[device_name].update({protocol+'_ip_addr': ip_addr})
        pass

    def display(self) -> None:
        """
        Display the graph and info graph
        """
        print(json.dumps(self.info_graph, indent=2))
        print("="*50)
        print(json.dumps(self.graph, indent=2))
        pass

    # After getting reply, the stream might be deleted
    def update_graph(self, reply:dict) -> None:
        """
        Based on returning information, update the content of graph. 
        Besides, for control consideration, activate and inactivate different streams.
        
        Active the stream only if the stream is on exchange some data
        Args:
            reply (dict): A dict structure which is exactly the same as graph which contains the observed information
        """
        for device_name, links in self.graph.items():
            for link_name, streams in list(links.items()):
                for port_name in list(streams.keys()):
                    try:
                        streams[port_name].update(reply[link_name][port_name])
                        self.info_graph[device_name][link_name][port_name].update({
                                                                                  'active': True})
                    except:
                        self.info_graph[device_name][link_name][port_name].update(
                            {'active': False})
        pass

    def graph_to_control_coefficient(self):
        """
        Function used to extract dict thru, throttle, mac from graph 

        Returns:
            thru (dict): A dict structured in {link_name:{}} containing the summation of estimated throughput over each stream on the link
            throttle (dict): A dict structured in {link_name:{}} containing the throttle value of each controllable link
            mcs (dict): A dict structured in {link_name:{}} containing the MCS value of each link
        """
        thru = {}
        throttle = {}
        mcs = {}
        # link throughput calculation and throttle detection
        for device_name, links in self.graph.items():
            for link_name, streams in links.items():
                link_thru = 0
                for port_number, stream in streams.items():
                    link_thru += stream['thru']
                    # detect file by keyword "file" in name or can be detected by thru value
                    if stream['thru'] == 0:
                        throttle.update({link_name: stream['thru']})
                thru.update({link_name: link_thru})
                mcs.update(
                    {link_name: self.info_graph[device_name][link_name]['MCS']})
        return thru, throttle, mcs

    # core function
    @staticmethod
    def _update_throttle(sorted_mcs:list, sorted_thru:list, allocated_times:list) -> list:
        """
        Compute throttle of each link.

        Args:
            sorted_mcs (list): the mcs value of each link in specific order where the order must match the order of sorted_thru 
            sorted_thru (list): the estimated throughput of each link (summation of estimated throughput of each stream)
            allocated_times (list): the time fraction determined from control algorithm

        Returns:
            sorted_throttle (list): the throttles to each link
        """
        # calculate the throughput/MCS (without file) of each link
        link_fraction = sum( [sorted_thru[i]/sorted_mcs[i] for i in range(len(sorted_mcs))] )
        # calculate the normalized throughput
        normalized_thru = ( link_fraction + allocated_times ) / len(sorted_mcs)
        return [normalized_thru * sorted_mcs[i] - sorted_thru[i] for i in range(len(sorted_mcs))]

    @staticmethod
    def _init_allocated_times(sorted_mcs:list, sorted_thru:list, init_factor:float):
        """
        Compute initiate fraction for latter control iteration

        Args:
            sorted_mcs (list): the mcs value of each link in specific order where the order must match the order of sorted_thru 
            sorted_thru (list): the estimated throughput of each link (summation of estimated throughput of each stream)
            init_factor (float): Heuristic value range from (0,1) used to get a conserved fraction

        Returns:
            fraction (float): the initialized fraction
        """
        allocated_times = 1 - sum( [sorted_thru[i]/sorted_mcs[i] for i in range(len(sorted_mcs))] )
        return allocated_times*init_factor

    def _link_to_port_throttle(self, link_throttle:dict) -> dict:
        """
        From each link throttle to port throttle by turning throttle value to each stream. 
        If more than one stream is controllable over one link, the throttle will be divided equally.

        Args:
            link_throttle (dict): {link_name:link_throttle_value}, throttle value of each link

        Returns:
            port_throttle (dict): {link_name: {stream_name: stream_throttle_value}}, throttle value of each stream
        """
        port_throttle = {}
        for device_name, links in self.graph.items():
            for link_name, streams in links.items():
                if link_name in link_throttle.keys():
                    port_throttle.update({link_name: {}})
                    # calculate file number
                    file_number = 0
                    for port_name, stream in streams.items():
                        if stream["thru"] == 0:
                            file_number += 1
                    # calculate port throttle
                    _port_throttle = link_throttle[link_name] / (file_number)
                    for port_name, stream in streams.items():
                        if stream["thru"] == 0:
                            stream.update({"throttle": _port_throttle})
                            port_throttle[link_name].update(
                                {port_name: _port_throttle})
        return port_throttle

    def update_throttle(self, fraction:float, reset_flag:bool) -> dict:
        """
        From computed fraction, computing the corresponding control to each file stream 

        Args:
            fraction (float): fraction of all the file stream
            reset_flag (bool): flag used to determine whether the reset happens

        Returns:
            port_throttle (dict): {link_name: {stream_name: stream_throttle_value}}, throttle value of each stream
        """
        # add last throttle value together
        thru, throttle, mcs = self.graph_to_control_coefficient()
        sorted_mcs = [mcs[key] for key in throttle]
        sorted_thru = [thru[key] for key in throttle]
        ##
        length = len(throttle)
        out_sorted_throttle = [float(x) for x in out_sorted_throttle]

        out_sorted_throttle = self._update_throttle([mcs[key] for key in throttle], [thru[key] for key in throttle], fraction)
        print("reset_flag",reset_flag)
        print("out_sorted_throttle",out_sorted_throttle)        
        for i, link_name in enumerate(throttle.keys()):
            throttle.update({link_name: out_sorted_throttle[i]})
        ##
        port_throttle = self._link_to_port_throttle(throttle)
        return port_throttle
