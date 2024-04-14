#!/usr/bin/env python3
import time
import subprocess as sp
from datetime import datetime
from wlsctrl.wlsctrl import set_tx_params, MmapContext

SHELL_RUN  = lambda x: sp.run(x, stderr=sp.PIPE, check=True, shell=True)

def timeit(func, *args, **kwargs):
    _t = time.time()
    func(*args, **kwargs)
    _t = time.time() - _t
    return _t

def file_transfer():
    SHELL_RUN('iperf3 -c 192.168.3.18 -p 5201 --tos 100 -n 100M')
    pass

def get_next_point():
    STEP = 1#minutes
    _t = datetime.now()
    next_minute = _t.minute + (STEP - _t.minute % STEP)
    next_hour   = _t.hour + (next_minute // 60)
    next_minute = next_minute % 60
    ##
    next_time =  datetime(_t.year, _t.month, _t.day, next_hour, next_minute)
    time_delta = (next_time - _t).total_seconds()
    return (next_time, time_delta)

CHOICES = [
    # (15, 15),
    # (15, 31),
    # (15, 63),
    # (15, 127),
    # (15, 511),
    # (15, 1023), #default
    # (31,  1023),
    (63,  1023),
    (127, 1023),
    (255, 1023),
    (511, 1023),
    (1023, 1023),
]

results = [ list() for _ in CHOICES ]

try:
    for i,params in enumerate(CHOICES):
        for k in range(5):
            # next_time, time_delta = get_next_point()
            # print(f'Next running point: {next_time}')
            ##
            print(f'set-{i}, tries {k}: running ...', end=' ', flush=True)
            with MmapContext() as ctx:
                set_tx_params(ctx, [2], -1, *params)
                _elapsed = timeit(file_transfer, )
                results[i].append( _elapsed )
            print(f'done. [{_elapsed}]')
except Exception as e:
    raise e
finally:
    print(results)
