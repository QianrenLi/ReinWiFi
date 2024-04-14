#!/usr/bin/env python3
import json
from tap import Connector

SYNC_CODE = lambda client, codebase: [client.sync_code(b) for b in codebase]
names = Connector().list_all()
clients = [ Connector(n) for n in names ]
## Default sync
with open('manifest.json') as f:
    manifest = json.load(f)
    codebase = manifest['codebase'].keys()

names = Connector().list_all()
clients = [ Connector(n) for n in names ]
default_code_base = ['manifest']
for c in clients:
    SYNC_CODE(c,default_code_base)
    c.reload()
    SYNC_CODE(c,codebase)
##
# exit()

