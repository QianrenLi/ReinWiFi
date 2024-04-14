import os
import argparse

def iperf_tx(target_ip, target_port, duration, mbps):
    cmd = "iperf3 -c %s -p %s -t %s -u -b %dM" % (target_ip, target_port, duration, mbps)
    run_cmd(cmd)

## run cmd
def run_cmd(cmd):
    try:
        os.system(cmd)
    except Exception as e:
        print("Error: %s" % e)


def main(args):
    iperf_tx(args.target_ip, args.target_port, args.duration, args.mbps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run iperf3 tx")
    parser.add_argument("-i", "--target_ip", type=str, default="192.168.3.35")
    parser.add_argument("-p", "--target_port", type=str, default="5201")
    parser.add_argument("-t", "--duration", type=str, default="10")
    parser.add_argument("-b", "--mbps", type=int, default=10)