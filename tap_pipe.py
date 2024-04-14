#!/usr/bin/env python3
import socket
import argparse
import json

class ipc_socket():

    def __init__(self, ip_addr, ipc_port, local_port=12345, link_name=""):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 196)
        self.sock.settimeout(1.5)
        self.sock.bind(("0.0.0.0", local_port))
        self.link_name = link_name
        self.ip_addr = ip_addr
        self.ipc_port = ipc_port

    # use socket to send udp to remote
    def send_udp(self, ipc_port, message):
        server_address = (self.ip_addr, ipc_port)
        message = json.dumps(message)
        self.sock.sendto(message.encode(), server_address)

    def send_cmd(self, *args):
        cmd = args[0]
        message = {"cmd": cmd}
        if cmd == "throttle":
            # extract port and tos
            message.update({"body": {}})
            message["body"].update(args[1])
        self.send_udp(self.ipc_port, message)

    def ipc_communicate(self, *args):
        self.send_cmd(*args)
        _buffer, addr = self.sock.recvfrom(2048)
        return _buffer
    def ipc_transmit(self, *args):
        self.send_cmd(*args)

    def close(self):
        self.sock.close()


def main(args):
    # create a ipc communicate socket
    sock = ipc_socket(args.ip_addr, args.port)
    # generate a request
    while True:
        print(sock.ipc_communicate('statistics',{}))
        sock.close()
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##
    parser.add_argument('-p', '--port', default=0,
                        type=int, help='binding port for receiving.')
    parser.add_argument('-i', '--ip_addr',
                        default="192.168.3.15", type=str, help='IP.')
    ##
    args = parser.parse_args()
    main(args)
