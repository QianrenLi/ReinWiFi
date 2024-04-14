#!/usr/bin/env python3
'''
{
    "get_avg_mcs": {
        "description": "Get average MCS with `iwconfig` together with small UDP stream.",
        "parameters": { "test_server":"", "duration":10 },
        "commands": [ "python3 record_mcs.py $test_server $duration" ],
        "outputs": {
            "mcs" : { "cmd":"echo $output_0", "format":".*" }
        }
    }
}
'''

import re
import socket
import subprocess as sp
import sys
import time
import threading

SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)
TX_RATE_FILTER = re.compile('Bit Rate=(.*) Mb')

INTERVAL = 0.001#s
TEST_SERVER = str(sys.argv[1])
MAX_TIME = int(sys.argv[2])#s

def test_udp():
    _addr = (TEST_SERVER, 12345)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b'@'*1024, _addr)

def get_tx_stat():
    iw_output = SHELL_RUN('iwconfig').stdout.decode()
    tx_rate    = float( TX_RATE_FILTER.findall(iw_output)[0] )
    return tx_rate

def main():
    threading.Thread(target=test_udp).start()
    ##
    init_time = time.time()
    results = list()
    ##
    while time.time() - init_time < MAX_TIME:
        tx_rate = get_tx_stat()
        results.append( tx_rate )
        time.sleep(INTERVAL)
    ##
    print( sum(results) / len(results) )

if __name__=='__main__':
    main()