#!/usr/bin/env python3
import sys
import ctypes
import argparse
import pkg_resources
from pathlib import Path

TX_PARAMS = {
    0 : {'aifs':2, 'cw_min':3,  'cw_max':7},
    1 : {'aifs':2, 'cw_min':7,  'cw_max':15},
    2 : {'aifs':3, 'cw_min':15, 'cw_max':1023},
    3 : {'aifs':7, 'cw_min':15, 'cw_max':1023}
}

class MmapContext:
    def __init__(self) -> None:
        _file = pkg_resources.resource_filename('wlsctrl', 'libwlsctrl.so')
        self.lib = ctypes.CDLL( _file )
        pass
    
    def __enter__(self):
        self.lib.w_init()
        return self.lib
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.lib.w_fini()
        pass
    pass

def reset_tx_params(ctx, if_ind = 0):
    ctx.set_tx_params(0, TX_PARAMS[0]['aifs'], TX_PARAMS[0]['cw_min'], TX_PARAMS[0]['cw_max'],if_ind)
    ctx.set_tx_params(1, TX_PARAMS[1]['aifs'], TX_PARAMS[1]['cw_min'], TX_PARAMS[1]['cw_max'],if_ind)
    ctx.set_tx_params(2, TX_PARAMS[2]['aifs'], TX_PARAMS[2]['cw_min'], TX_PARAMS[2]['cw_max'],if_ind)
    ctx.set_tx_params(3, TX_PARAMS[3]['aifs'], TX_PARAMS[3]['cw_min'], TX_PARAMS[3]['cw_max'],if_ind)
    pass

def set_tx_params(ctx, acq:list, aifs:int, cw_min:int, cw_max:int, if_ind = 0):

    for ac in acq:
        aifs   = aifs   if aifs>=0   else TX_PARAMS[ac]['aifs']
        cw_min = cw_min if cw_min>=0 else TX_PARAMS[ac]['cw_min']
        cw_max = cw_max if cw_max>=0 else TX_PARAMS[ac]['cw_max']
        ret = ctx.set_tx_params(ac, aifs, cw_min, cw_max, if_ind)
        # print(f'set AC{ac}: {ret}.')
    pass

def execute(command, args):
    with MmapContext() as ctx:
        if command=='reset':
            reset_tx_params(ctx)
        elif command=='set':
            acq = [0,1,2,3] if args.ac=='all' else [int(args.ac)]
            set_tx_params(ctx, acq, args.aifs, args.cw_min, args.cw_max)
        else:
            print(f'Nothing happened.')
    pass

def main():
    parser = argparse.ArgumentParser(description='wlsctrl for wlsops_hack.')
    subparsers = parser.add_subparsers(dest='command')
    ##
    ac_set = subparsers.add_parser('set', help='set specific AC queue EDCA parameters.')
    ac_set.add_argument('ac',     type=str, help='AC queue index, "all" to choose all.')
    ac_set.add_argument('aifs',   type=int, help='0 - 255')
    ac_set.add_argument('cw_min', type=int, help='0 - 65535')
    ac_set.add_argument('cw_max', type=int, help='0 - 65535')
    ##
    ac_reset = subparsers.add_parser('reset', help='reset specific AC queue with default EDCA parameters.')
    ac_reset.add_argument('ac',   type=str, help='AC queue index, "all" to choose all.')
    ##
    args = parser.parse_args()
    execute(args.command, args)
    pass

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        raise e
    finally:
        pass
