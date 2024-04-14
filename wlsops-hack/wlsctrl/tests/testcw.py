#!/usr/bin/env python3
import time
import subprocess as sp
from datetime import datetime
from wlsctrl.wlsctrl import set_tx_params, MmapContext,reset_tx_params

try:
    with MmapContext() as ctx:
        set_tx_params(ctx, [0], -1, 63,1023)
        print("CW changed")
except Exception as e:
    raise e
