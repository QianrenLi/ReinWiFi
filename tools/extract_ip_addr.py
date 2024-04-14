#!/usr/bin/env python3
import re
import sys
import psutil

# path = sys.argv[1]
# keyword = sys.argv[2]

## init ip information
def get_ip():
    addrs = psutil.net_if_addrs()
    ## from addrs to info name
    info = []
    for key in addrs.keys():
        for addr in addrs[key]:
            if addr.family == 2:
                info.append((key, addr.address))
                # info[key] = addr.address
    return info

# def p2p_wlan_extract(terminal_output, keyword):
#     pattern_str = "(%s).*?([\d]+\.[\d]+\.[\d]+\.[\d]+)" % keyword
#     p2p_wlan_pattern = re.compile(pattern_str)
#     matches = re.findall(p2p_wlan_pattern, repr(terminal_output))
#     # Extract the matched keywords and IPv4 addresses
#     if matches:
#         return matches # Example: [('p2p', '127.0.0.1'), ('wlan', '10.26.42.138')]
#         raise Exception("Not exist a valid ip")


# with open(path,'r') as f:
#     output_template = f.read()


# print(repr(output_template))

    
# print(p2p_wlan_extract(output_template, keyword))
print(get_ip())