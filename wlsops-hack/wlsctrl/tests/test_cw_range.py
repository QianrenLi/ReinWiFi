#!/usr/bin/env python3
import sys, time
import subprocess as sp
from wlsctrl.wlsctrl import set_tx_params, MmapContext

SHELL_RUN  = lambda x: sp.run(x, stderr=sp.PIPE, check=True, shell=True)

IPERF_ADDR = '192.168.137.1'
IPERF_PORT = 5201

USE_UDP    = '-u -b 0'
TEST_BLOCK = '-n 100M'
TEST_TIME  = '-t 30'

def set_ac2_cw_min(cw_min):
    with MmapContext() as ctx:
        set_tx_params(ctx, [2], -1, cw_min, -1)

def run_ac0_default():
    SHELL_RUN(f'iperf3 -c {IPERF_ADDR} -p {IPERF_PORT} {USE_UDP} {TEST_TIME} --tos 200')

def run_ac2_default():
    SHELL_RUN(f'iperf3 -c {IPERF_ADDR} -p {IPERF_PORT} {USE_UDP} {TEST_TIME} --tos 100')

def run_ac2_with_cw_min(cw_min:int):
    assert( cw_min in range(1,1024) )
    set_ac2_cw_min(cw_min)
    run_ac2_default()
    ##
    time.sleep(1)
    set_ac2_cw_min(-1)
    pass

if __name__=='__main__':
    __usage = f'Usage: {__file__} ac0/ac2 <cw_min>'
    ##
    if len(sys.argv)<2:
        print(__usage)
    elif sys.argv[1]==['ac2'] and len(sys.argv)<3:
        print(__usage)
    else:
        if sys.argv[1]=='ac0':
            run_ac0_default()
        elif sys.argv[1]=='ac2':
            cw_min = int(sys.argv[2])
            run_ac2_with_cw_min(cw_min)
        else:
            print(__usage)
    pass

