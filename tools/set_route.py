import psutil
from ipaddress import ip_network
import argparse
import os

## Setup route table for Intel card and Realtek card
def two_IC_setup():
    info = init_info()
    if_name_Intel = "wlp"
    if_name_Realtek = "wlx"
    rets = []
    rets.append(route_setup(info, if_name_Intel, 111))
    rets.append(route_setup(info, if_name_Realtek, 112))
    if -1 in rets:
        print("Setup Failed")
        return None
    else:
        print("Setup Successful")
        return [if_name_Intel, if_name_Realtek]

def subset_IC_setup():
    info = init_info()
    subnet = "192.168"
    rets = []
    if_name = []
    for key in info:
        table_id = 111
        if subnet in info[key]:
            rets.append(route_setup(info, key, table_id))
            if_name.append(key)
            table_id += 1
    
    if -1 in rets or len(if_name) == 0:
        print("Setup Failed")
        return None
    else:
        print("Setup Successful")
        return if_name


## init ip information
def init_info():
    addrs = psutil.net_if_addrs()
    ## from addrs to info name
    info = {}
    for key in addrs.keys():
        for addr in addrs[key]:
            if addr.family == 2:
                info[key] = addr.address
    return info

## init phy Interface Information
def init_phy_info():
    ## get phy - interface by iw dev
    phy_if = {}
    cmd = "iw dev"
    ret = os.popen(cmd).read()
    reGex  = "phy#(\d+)\s+Interface\s+(\S+)"
    ## search ret for phy and interface in gm
    import re
    match = re.findall(reGex, ret)
    for m in match:
        phy_if[m[1]] = m[0]
    return phy_if

def conf_wpa_supplicant_one_NIC(ip_addr:int=15):
    phy_if = init_phy_info()
    if len(phy_if) != 1:
        print("Interface Number not equal to 1")
        return -1
    else:
        if_name = list(phy_if.keys())[0]
        conf_wpa_supplicant(if_name, ip_addr)
        print("Setup Config Successfully")
        return if_name

## config connection
def conf_wpa_supplicant(if_name:str, index:int):
    path = "/etc/wpa_supplicant/experiment-%s.conf" % if_name
    if not os.path.exists(path):
        os.mknod(path)
        print("Create file successful")
        ## write config to file
        with open(path, "w") as f:
            f.write("network={\n    ssid=\"HONOR-TEST-AP_5G\"\n    key_mgmt=NONE\n}\n")
        print("Setup Config Successfully")
    else:
        print("File Exist")
    stop_wpa_supplicant(if_name)
    print("Stop Successfully")
    start_wpa_supplicant(if_name, index)
    print("Start Successfully")

def start_wpa_supplicant(if_name:str, index:int):
    path = "/etc/wpa_supplicant/experiment-%s.conf" % if_name
    cmd = "sudo wpa_supplicant -B -i %s -c %s" % (if_name, path)
    cmd += "; sudo dhcpcd -S ip_address=192.168.3.%d/24 -S routers=192.168.3.1 -S domain_name_servers=192.168.3.1 %s" % (index, if_name)
    run_cmd(cmd)

def stop_wpa_supplicant(if_name:str):
    cmd = "sudo wpa_cli -i %s terminate" % if_name
    run_cmd(cmd)

def create_name_space(namespace:str, if_name:str):
    phy_if = init_phy_info()
    if if_name in phy_if.keys():
        _create_name_space(namespace, f'phy{phy_if[if_name]}', if_name)
    else:
        print("Interface not found")


## config name space
def _create_name_space(namespace:str, phy: str, if_name:str):
    cmd = "sudo ip netns del %s ;" % namespace
    cmd += "sudo ip netns add %s ;" % namespace
    cmd += add_phy_to_namespace(namespace, phy, if_name)
    cmd += "; " + into_name_space(namespace)
    run_cmd(cmd)

def add_phy_to_namespace(namespace:str, phy:str, if_name:str) -> str:
    cmd = "sudo iw phy %s set netns name %s" % (phy, namespace)
    cmd += "; sudo ip netns exec %s ip link set %s up" % (namespace, if_name)
    return cmd

def into_name_space(namespace:str):
    cmd = "sudo nsenter --net=/run/netns/%s" % namespace
    return cmd

## Setup route table
def route_setup(info, if_type, table_id, netmask=24):
    for if_name in info.keys():
        if if_type in if_name:
            ip_addr = ip_network(info[if_name] + "/" + str(netmask), strict=False)
            # get default first host as gateway from ip_addr
            gateway = str(ip_addr[1])
            cmd = "sudo ip route add " + str(ip_addr) + " dev " + if_name + " proto kernel scope link src " + info[if_name] + " table " + str(table_id)
            cmd = cmd + " && sudo ip route add default via " + gateway + " table " + str(table_id)
            cmd = cmd + " && sudo ip rule add from " + info[if_name] + " table " + str(table_id)
            # Execute cmd in bash
            run_cmd(cmd)
            return 1
    return -1

## Subprocess run cmd
def run_cmd(cmd):
    os.system(cmd)

def sys_conf_init(if_names):
    if if_names is None:
        print("Config do not write")
        return -1
    path = "/etc/sysctl.conf"
    info = init_info()
    if not if_conf_exist(path):
        write_sys_conf(info, path, if_names)
    else:
        print("Config Exists")

def sys_conf_exit(line_num):
    path = "/etc/sysctl.conf"
    if if_conf_exist(path):
        remove_last_k_lines(path, line_num)
    else:
        print("Config not exists")

def if_conf_exist(path):
    with open(path, "r") as f:
        for line in f.readlines():
            if "net.ipv4.conf.all.all_announce = 2" in line:
                return True
    return False

## Write to /etc/sysctl.conf
def write_sys_conf(info, path, if_names):
    print(if_names)
    if None not in if_names:
        print("start write")
        with open(path, "a") as f:
            f.write("\nnet.ipv4.conf.all.all_announce = 2\n")               #in remove last k lines function, it will additionally remove "\n"
            f.write("net.ipv4.conf.all.arp_ignore = 1\n")
            f.write("net.ipv4.conf.default.accept_source_route = 0\n")
            f.write("net.ipv4.conf.default.arp_announce = 2\n")
            f.write("net.ipv4.conf.default.arp_ignore = 1\n")
            f.write("net.ipv4.conf.default.rp_filter = 1\n")
            for if_name in if_names:
                f.write("net.ipv4.conf.%s.arp_announce = 2\n" % if_name)
                f.write("net.ipv4.conf.%s.arp_ignore = 1\n" % if_name)
        print("Lines written:\t", 6 + 2*len(if_names))
    else:
        print("write failed")

def remove_last_k_lines(path, k):
    with open(path, "r+") as f:
        f.seek(0, 2)

        pos = f.tell()
        for count in range(k):
            pos -= 1
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, 0)

        if pos > 0:
            f.seek(pos, 0)
            f.truncate()
            print("Remove last " + str(k) + " lines")
        else:
            print("Remove fails -- not enough lines")

def main(args):
    if args.start:
        create_name_space(args.namespace, args.ifname)
    else:
        conf_wpa_supplicant_one_NIC(args.ip)


## Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--start", action="store_true", help = "Start config and cmd setup"
    )
    parser.add_argument(
        "--exit", action = "store_true", help  = "Exit config"
    )
    parser.add_argument(
        "-w", "--wpa", action="store_true", help = "Start wpa connection" 
    )
    parser.add_argument(
        "-l", "--lines", type= int, help="lines required to be removed"
    )
    parser.add_argument(
        "-n", "--namespace", type = str, help="name of namespace"
    )
    parser.add_argument(
        "-p", "--phy", type= str, help= "name of the phy layer"
    )
    parser.add_argument(
        "-i", "--ifname", type= str, help= "name of the interface" 
    )
    parser.add_argument(
        "--ip", type= int, help= "last 8 bits of ipv4"
    )

    args = parser.parse_args()
    main(args)


    # test_cmd = "ls"
    # subprocess.run(test_cmd, shell=True)
    # remove_last_k_lines("test.txt", 3)
    # sys_conf_init()
    # sys_conf_exit()