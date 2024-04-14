#!/usr/bin/env python3
import sys

port = int(sys.argv[1])
tos = int(sys.argv[2])
path = "../stream-replay/logs/rtt-%d@%d.txt" % (port, tos)

with open(path, 'r') as f:
    lines = f.readlines()

rtt_list = []
for line in lines:
    seq = int(line.split(' ')[0])
    rtt_list.append(float(line.split(' ')[1]))

print(sum(rtt_list)/len(rtt_list))
