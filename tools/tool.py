#!/usr/bin/env python3
import sys
import json

_template = {
    'type': 'UDP',
    'npy_file': '75MBps.npy',
    'tos': 100,
    'port': 5201,
    'throttle': 0,
    "priority": "",
    'calc_rtt': False,
    'no_logging': False,
    'start_offset': 0,
    'duration':[0,10]
}
_content = {
    "window_size": 500,
    'streams': []
}


NAME_MAP = {
    'file_port': 'file_75MB.npy',
    'proj_port': '4Xh2h.npy',
    'real_port': '7_50MBps.npy'
}

if len(sys.argv) not in [4, 5, 11]:
    print(len(sys.argv))
    print(sys.argv)
    print('Usage: python3 tool.py <key> <num> or python3 tool.py <key> <idx> <value>.')
    exit()

path = '../stream-replay/data/%s' % sys.argv[1]

if len(sys.argv) == 4:
    key = sys.argv[2]
    if key == 'reset':
        idx = int(sys.argv[3])
        content = _content
        content['streams'] = [_template for _ in range(idx)]
        with open(path, 'w+') as f:
            json.dump(content, f, indent=4)
        exit()
    elif key == 'throttle':
        argument = float(sys.argv[3])
        file_idx = []
        with open(path, 'r') as f:
            content = json.load(f)
        for _idx in range(len(content['streams'])):
            file_idx.append(_idx) if content['streams'][_idx]['npy_file'] == NAME_MAP['file_port'] else None
        for _idx in file_idx:
            content['streams'][_idx][key] = argument/len(file_idx)
        with open(path, 'w') as f:
            json.dump(content, f, indent=4) 
    else:
        raise Exception(f'wrong key name: {key}.')
elif len(sys.argv) == 5:
    with open(path, 'r') as f:
        content = json.load(f)
    ##
    key = sys.argv[2]
    idx = int(sys.argv[3])
    ##

    if key == 'throttle':
        argument = float(sys.argv[4])
    elif key == 'port':
        argument = int(sys.argv[4])
    elif key == 'tos':
        argument = int(sys.argv[4])
    elif key == 'npy_file':
        argument = sys.argv[4]
    elif key == 'priority':
        _arg = int(sys.argv[4])
        if _arg == 1:
            content.update({"orchestrator":""})
        argument = "" if _arg == 0 else "guarded,"
    else:
        raise Exception(f'wrong key name: {key}.')
    ##
    if idx <= len(content['streams']):
        content['streams'][idx][key] = argument
    ##
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)
    ##
elif len(sys.argv) == 11:
    with open(path, 'r') as f:
        content = json.load(f)
    ##
    idx = int(sys.argv[2])
    ##
    if idx <= len(content['streams']):
        content['streams'][idx]['port'] = int(sys.argv[3])
        content['streams'][idx]['tos'] = int(sys.argv[4])
        content['streams'][idx]['npy_file'] = sys.argv[5]
        content['streams'][idx]['calc_rtt'] = eval(sys.argv[6])
        content['streams'][idx]['no_logging'] = eval(sys.argv[7])
        content['streams'][idx]['duration'] = [float(sys.argv[8]),float(sys.argv[9])]
        content['streams'][idx]['throttle'] = float(sys.argv[10])
        # print(eval(sys.argv[8]))

    ##
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)
else:
    raise Exception('wrong number of arguments.')