#!/usr/bin/env python3
import time
import subprocess as sp
from datetime import datetime
from wlsctrl.wlsctrl import set_tx_params, MmapContext,reset_tx_params

try:
    with MmapContext() as ctx:
        reset_tx_params(ctx)
        print("reset")
except Exception as e:
    raise e
