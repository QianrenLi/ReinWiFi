#!/usr/bin/env python3
from abc import abstractmethod
import argparse
import ipaddress
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import string
import struct
import subprocess as sp
from tempfile import NamedTemporaryFile
import threading
import time
import traceback
from queue import Queue

SERVER_PORT = 11112
IPC_PORT    = 52525
CHUNK_SIZE  = 4096
BUFFER_SIZE = 10240

GEN_TID = lambda: ''.join([random.choice(string.ascii_letters) for _ in range(8)])
SHELL_POPEN = lambda x: sp.Popen(x, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)

class KeyError(Exception): pass #override `KeyError`
class StdErrException(Exception): pass
class TimeoutException(Exception): pass
class NoResponseException(Exception): pass
class InvalidRequestException(Exception): pass
class AutoDetectFailureException(Exception): pass
class ClientConnectionLossException(Exception): pass
class ClientNotFoundException(Exception): pass
class CodebaseNonExistException(Exception): pass

class UntangledException(Exception):
    def __init__(self, args):
        err_cls, err_msg = args
        # err_msg = f'\b:{err_cls} {err_msg}'
        raise eval(err_cls)(err_msg)

    @staticmethod
    def format(role:str, e:Exception):
        err_cls = type(e).__name__
        err_msg = '\b:[[{role}]]: {err}\n{tb}'.format(
            role=role, err=str(e), tb=traceback.format_exc() )
        return (err_cls, err_msg)
    pass

def _fixed_recv(sock:socket.socket, length):
    data = sock.recv(length)
    while len(data) < length:
        remains = sock.recv(length - len(data))
        if not remains: break
        data += remains
    return data

def _recv(sock:socket.socket):
    _len = struct.unpack('I', _fixed_recv(sock,4))[0]
    _msg = _fixed_recv(sock,_len)
    return _msg

def _send(sock:socket.socket, msg):
    if isinstance(msg, bytes):
        _msg = msg
    elif isinstance(msg, str):
        _msg = msg.encode()
    else:
        _msg = json.dumps(msg).encode()
    ##
    _len = struct.pack('I', len(_msg))
    _msg = _len + _msg
    sock.send(_msg)
    pass

def _sync(sock:socket.socket, msg):
    _send(sock, msg)
    _msg = _recv(sock).decode()
    _msg = json.loads(_msg)
    return _msg

def _send_file(sock:socket.socket, name:str, file_glob:str) -> None:
    file_list = Path(__file__).parent.resolve().glob(file_glob)
    file_list = filter(lambda x:x.is_file(), file_list)
    ##
    for _file in file_list:
        file_name = _file.relative_to( Path('.').resolve() ).as_posix()
        print(f'Send to {name}: "{file_name}" ... ', end='', flush=True)
        with open(_file, 'rb') as fd:
            fd.seek(0, os.SEEK_END)
            file_len = fd.tell()
            fd.seek(0, 0)
            trunk_num = file_len // CHUNK_SIZE
            ## (1) send file name
            _send(sock, file_name)
            ## (2) send chunks
            for i in range(trunk_num):
                _send(sock, fd.read(CHUNK_SIZE))
            ## (3) send remains
            _send(sock, fd.read())
            _send(sock, '@end') #finalize file sending
        print('done.')
    _send(sock, '') #finalize sending
    pass

def _recv_file(sock:socket.socket, file_glob:str) -> None:
    try:
        sock.settimeout(1.0)
        while True:
            with NamedTemporaryFile() as fd:
                ## (1) recv file name
                file_name = _recv(sock).decode()
                if not file_name: break
                ## (2) recv chunks until end
                while True:
                    _chunk = _recv(sock)
                    if _chunk==b'@end': break
                    fd.write(_chunk)
                fd.flush()
                ## (3) copy to codebase
                if Path(file_name).match(file_glob):
                    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fd.name, file_name)
                    print(f'"{file_name}" received.')
                else:
                    print(f'"{file_name}" rejected.')
    except Exception as e:
        sock.settimeout(None)
        raise e
    pass

def _extract(cmd:str, format:str):
    try:
        ret = SHELL_RUN(cmd).stdout.decode()
    except sp.CalledProcessError as e:
        raise StdErrException( e.stderr.decode() )
    else:
        ret = re.findall(format, ret)
        ret = [ x for x in ret if x ]
        if len(ret)==0: ret=''
        if len(ret)==1: ret=str(ret[0])
    return ret

def _execute(name, task_pool, tid, config, params, timeout) -> None:
    try:
        timeout = timeout if timeout>=0 else 999
        exec_params = config['parameters'] if 'parameters' in config else {}
        exec_params.update(params)
        ##
        commands = config['commands'].copy() if 'commands' in config else []
        for i in range(len(commands)):
            for k,v in exec_params.items():
                commands[i] = commands[i].replace(f'${k}', str(v))
        ##
        processes = [ SHELL_POPEN(cmd) for cmd in commands ]
        returns = [ proc.poll() for proc in processes ]
        _now = time.time()
        while None in returns and time.time() - _now < timeout:
            returns = [ proc.poll() for proc in processes ]
            time.sleep(0.001)
        ##
        err = list()
        for i,ret in enumerate(returns):
            if ret is None:
                processes[i].kill()
                processes[i].communicate()
                err.append( TimeoutException(f'{name}, [{i}]-th command.') )
            ##
            if ret != 0:
                _stderr = processes[i].stderr; assert(not _stderr==None)
                err.append( StdErrException(_stderr.read().decode()) )
        if err: raise err[0] #raise the first error
        ##
        outputs = dict()
        for i,p in enumerate(processes):
            _stdout = p.stdout; assert(not _stdout==None)
            outputs.update({ f'$output_{i}' : repr(_stdout.read().decode().strip())  })
        print(outputs)
        ##
        results = dict()
        output_items = config['outputs'].items() if 'outputs' in config else dict()
        for key,value in output_items:
            cmd, _format = value['cmd'], value['format']
            for k,o in outputs.items():
                cmd = cmd.replace(k,o)
            for k,v in exec_params.items():
                cmd = cmd.replace(f'${k}', str(v))
            results[key] = _extract(cmd, _format)
    except Exception as e:
        task_pool[tid]['results'] = { 'err': UntangledException.format('Client', e) }
    else:
        task_pool[tid]['results'] = results
    pass

class Request:
    def __init__(self, handler):
        self.handler = handler

    def console(self, args, client='') -> dict:
        '''Default console behavior: [console] <--(bypass)--> [server].'''
        ## --> [server]
        _request = type(self).__name__
        args = {'request':_request, 'args':args}
        client = client if client else self.handler.client
        req = '{request} {client}@{args}'.format(
                request=_request, client=client, args=json.dumps(args) ).encode()
        self.handler.sock.send(req)
        ## <-- [server]
        res = self.handler.sock.recv(BUFFER_SIZE).decode()
        res = json.loads(res)
        if 'err' in res:
            UntangledException(res['err'])
        return res

    def server(self, args:str) -> dict:
        '''Default server behavior: [server] <--(bypass)--> [proxy].'''
        _request = type(self).__name__
        name, args = args.split('@', maxsplit=1)
        ## handle at server's side
        if name in ['', self.handler.name]:
            req = { '__server_role__':'server', 'args':json.loads(args)['args'] }
            return req
        ## else bypass to proxy
        try:
            client = self.handler.client_pool[name]
        except:
            e = ClientNotFoundException(f'Client "{name}" not exists.')
            res = { 'err': UntangledException.format('Server', e) }
        else:
            ## --> [proxy]
            client['tx'].put((_request, args))
            ## <-- [proxy]
            res = client['rx'].get()
        return res

    def proxy(self, conn, _name:str, _task_pool:dict, args:str) -> dict:
        '''Default proxy behavior: [proxy] <--(bypass)--> [client].'''
        return _sync(conn, args)

    @abstractmethod
    def client(self, args:dict) -> dict: pass
    pass

class Handler:
    ## console <--> server <--> proxy <--> client
    def handle(self, request:str, args, **kwargs) -> dict:
        try:
            handler = getattr(Handler, request)(self)
        except:
            raise InvalidRequestException(f'Request "{request}" is invalid.')
        ##
        if isinstance(self,MasterDaemon):
            return handler.server(args, **kwargs)
        if isinstance(self,SlaveDaemon):
            return handler.client(args, **kwargs)
        if isinstance(self,Connector):
            return handler.console(args, **kwargs)
        return dict()

    def proxy(self, name:str, client:dict, request:str, args:str) -> dict:
        handler = getattr(self, request)(self)
        conn, task_pool = client['conn'], client['task_pool']
        return handler.proxy(conn, name, task_pool, args)

    class list_all(Request):
        def server(self, _args):
            return  { k:v['addr'] for k,v in self.handler.client_pool.items() }
        pass

    class describe(Request):
        def server(self, args):
            req = super().server(args)
            res = self.client(req['args']) if '__server_role__' in req else req
            return res
        def client(self, _args):
            return { k:v['description'] for k,v in self.handler.manifest['functions'].items() }
        pass

    class reload(Request):
        def server(self, args):
            req = super().server(args)
            res = self.client(req['args']) if '__server_role__' in req else req
            return res
        def client(self, _args):
            self.handler._reload()
            return {'res':True}

    class info(Request):
        def client(self, args):
            function = args['function']
            return self.handler.manifest['functions'][function]
        def server(self, args):
            req = super().server(args)
            res = self.client(req['args']) if '__server_role__' in req else req
            return res
        pass

    class execute(Request):
        def server(self, args):
            req = super().server(args)
            res = self.client(req['args']) if '__server_role__' in req else req
            return res

        def client(self, args):
            params, timeout, fn = args['parameters'], args['timeout'], args['function']
            config = self.handler.manifest['functions'][fn]
            ##
            tid = GEN_TID()
            _thread = threading.Thread(target=_execute, args=(self.handler.name, self.handler.task_pool, tid, config, params, timeout))
            self.handler.task_pool[tid] = { 'handle':_thread }
            _thread.start()
            return { 'tid': tid }
        pass

    class batch_execute(Request):
        def console(self, args:list):
            ## --> [server]
            req_args = '##'.join([ f'{x[0]}@{x[1]}' for x in args ])
            req = f'batch_execute {req_args}'.encode()
            self.handler.sock.send(req)
            ## <-- [server]
            res = self.handler.sock.recv(BUFFER_SIZE).decode()
            res = json.loads(res)
            if 'err' in res:
                UntangledException(res['err'])
            return res

        def server(self, args: str) -> dict:
            _handler = Handler.execute(self.handler)
            results = list()
            ##
            arguments = [x.split('@') for x in args.split('##')]
            for (name, args) in arguments:
                if name in ['', self.handler.name]:
                    p_args = json.loads(args)['args']
                    res = _handler.client(p_args)
                    res.setdefault('err'); res.setdefault('tid') # type: ignore
                else:
                    try:
                        client = self.handler.client_pool[name]
                    except Exception as e:
                        res = { 'tid':None, 'err': UntangledException.format('Server', e) }
                    else:
                        client['tx'].put(('execute', args)) ## --> [proxy]
                        res = { 'tid':'',   'err':None }
                results.append(res)
            ##
            for i,(name,_) in enumerate(arguments):
                if results[i]['tid']=='':
                    client = self.handler.client_pool[name]
                    res = client['rx'].get()  ## <-- [proxy]
                    res.setdefault('err'); res.setdefault('tid') # type: ignore
                    results[i] = res
            ##
            results = {'tid_list': [item['tid'] for item in results], 'err_list': [item['err'] for item in results]}
            return results

        pass

    class fetch(Request):
        def server(self, args):
            req = super().server(args)
            res = self.client(req['args']) if '__server_role__' in req else req
            return res

        def client(self, args):
            tid = args['tid']
            if 'results' in self.handler.task_pool[ tid ]:
                res = self.handler.task_pool[ tid ]['results']
            else:
                raise NoResponseException(f'"{self.handler.name}", tid={tid}.')
            self.handler.task_pool[ tid ]['handle'].join()
            return res
        pass

    class sync_code(Request):
        def proxy(self, conn, name: str, _task_pool: dict, args: str) -> dict:
            res = super().proxy(conn, name, _task_pool, args)
            if 'err' in res:
                return res
            ##
            p_args = json.loads(args)['args']
            basename = p_args['basename']
            codebase = self.handler.manifest['codebase']
            if basename not in codebase:
                raise CodebaseNonExistException(basename)
            ##
            file_glob = codebase[basename]
            _send_file(conn, name, file_glob)
            return {'res':True}

        def client(self, args: dict) -> dict:
            basename = args['basename']
            codebase = self.handler.manifest['codebase']
            if basename not in codebase:
                raise CodebaseNonExistException(basename)
            else:
                _send(self.handler.sock, {'res':True})
            ##
            file_glob = codebase[basename]
            _recv_file(self.handler.sock, file_glob)
            return {'res':True}
        pass

    pass

class SlaveDaemon(Handler):
    def __init__(self, port:int, manifest:dict, addr='', alt_name=''):
        client_name = alt_name if alt_name else manifest['name']
        self.name = client_name if client_name else f'client-{GEN_TID()}'
        self.manifest = manifest
        ##
        self.addr, self.port = addr, port
        self.task_pool = dict()
        pass

    def _reload(self):
        manifest = open('./manifest.json')
        manifest = json.load( manifest )
        self.manifest = manifest
        print(f'{time.ctime()}: Manifest reloaded.')
        pass

    def auto_detect(self) -> socket.socket:
        ## get default gateway
        o = SHELL_RUN('ip route | grep default').stdout.decode()
        iface_name = re.findall('default via (\\S+) dev (\\S+) .*', o)[0][1]
        o = SHELL_RUN('ip addr').stdout.decode()
        iface_network = re.findall(f'.+inet (\\S+).+{iface_name}', o)[0]
        all_hosts = ipaddress.IPv4Network(iface_network, strict=False).hosts()
        ## auto detect brutally
        print(f'Auto-detect master over {iface_name} ...')
        for _,host in enumerate(all_hosts):
            addr = str(host)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            if sock.connect_ex((addr, self.port)) == 0:
                print(f'Find master on {addr}')
                sock.settimeout(None)
                return sock
        raise AutoDetectFailureException('No master found.')

    def daemon(self, sock):
        while True:
            try:
                msg = json.loads( _recv(sock).decode() )
                res = self.handle(msg['request'], msg['args'])
            except Exception as e:
                err = { 'err': UntangledException.format('Client', e) }
                _send(sock, err)
            else:
                _send(sock, res)
        pass

    def start(self):
        if self.addr:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.addr, self.port))
        else:
            self.sock = self.auto_detect()
        ## initial register
        _send(self.sock, {'name':self.name})
        print( f'Client "{self.name}" is now on.' )
        ##
        self.daemon(self.sock)
        pass

    pass

class MasterDaemon(Handler):
    def __init__(self, port:int, ipc_port:int, manifest={}):
        self.name = ''
        self.manifest = manifest
        ##
        self.port, self.ipc_port = port, ipc_port
        self.client_pool = dict()
        self.task_pool = dict()
        pass

    def _reload(self):
        manifest = open('./manifest.json')
        manifest = json.load( manifest )
        self.manifest = manifest
        print(f'{time.ctime()}: Manifest reloaded.')
        pass

    def proxy_service(self, name, tx:Queue, rx:Queue):
        while True:
            try:
                res = self.proxy( name, self.client_pool[name], *rx.get() )
            except struct.error:
                e = ClientConnectionLossException(f'{name} disconnected.')
                tx.put({ 'err': UntangledException.format('Proxy', e) })
                self.client_pool.pop(name)
                break
            except Exception as e:
                err = { 'err': UntangledException.format('Proxy', e) }
                tx.put( err )
            else:
                tx.put( res )
        pass

    def daemon(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.ipc_port))
        ##
        print('IPC Daemon is now on.')
        while True:
            msg, addr = sock.recvfrom(BUFFER_SIZE)
            cmd, args = msg.decode().split(maxsplit=1)
            ##
            try:
                res = self.handle(cmd, args)
                res = json.dumps(res).encode()
            except Exception as e:
                err = { 'err': UntangledException.format('Server', e) }
                err = json.dumps(err).encode()
                sock.sendto(err, addr)
            else:
                sock.sendto(res, addr)
            pass
        pass

    def serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', self.port))
        sock.listen()
        ##
        print(f'Server is now on.')
        while True:
            conn, addr = sock.accept()
            try:
                name = json.loads( _recv(conn).decode() )['name']
                print(f'Client "{name}" connected.')
            except:
                print(f'malicious connection detected: {addr}.')
            else:
                tx, rx = Queue(), Queue()
                handler = threading.Thread(target=self.proxy_service, args=(name, rx, tx))
                self.client_pool.update({
                    name:{'handler':handler,'conn':conn,'task_pool':{},'addr':addr,'tx':tx,'rx':rx} })
                handler.start()
        pass

    def start(self):
        self.server_thread = threading.Thread(target=self.serve)
        self.server_thread.start()
        self.daemon()
        pass

    pass

class Connector(Handler):
    """The IPC broker used to communicate with the tap server, via UDP.

    Args:
        client (str): The client name. Leave empty to only query from server.
        addr (str): (Optional) Specify the IP address of the server, default as ''.
        port (int): (Optional) Specify the port of the server, default as 52525.
    """

    class BatchExecutor:
        __slots__ = ['parent', 'pipeline', 'task_list', 'tid_list', 'outputs']
        def __init__(self, parent):
            self.parent = parent
            self._initialize()
            pass

        def _initialize(self):
            self.pipeline, self.task_list = list(), list()
            self.tid_list, self.outputs = list(), list()
            pass

        def _apply_tasks(self):
            res = self.parent.handle('batch_execute', self.task_list)
            [ UntangledException(e) for e in res['err_list'] if e ]
            ##
            res = res['tid_list']
            res = [ (name,tid) for (name,_),tid in zip(self.task_list, res) ]
            self.tid_list.extend( res )
            self.task_list = list() #cleanup
            pass

        def _apply_fetch(self):
            res = [ self.parent.handle('fetch', {'tid':tid}, client=name) if tid else None
                        for (name,tid) in self.tid_list ]
            self.outputs.extend( res )
            self.tid_list = list() #cleanup

        def _apply_outputs(self):
            if self.task_list: self._apply_tasks()
            outputs = self.outputs
            self._initialize() #cleanup
            return outputs

        def batch(self, client:str, function:str, parameters:dict={}, timeout:float=-1):
            """Batch execution by simultaneously sending commands, then use `.apply` to apply send action.

            Args:
                client (str): The client name (or the server name).
                function (str): The function names.
                parameters (dict): The parameters provided for the function. The absent values will use the default values in the manifest.
                timeout (float): The longest time in seconds waiting for the outputs from function execution.

            Returns:
                Self: used for chain call.
            """
            args = {'function':function, 'parameters':parameters, 'timeout':timeout}
            args = {'request':'execute', 'args':args}
            cmd = ( client, json.dumps(args) )
            self.pipeline.append(cmd)
            return self

        def batch_all(self, task_list:list):
            """Batch all the executions in list format.

            Args:
                task_list (list): the executions in list format.

            Returns:
                Self: used for chain call.
            """
            for task in task_list:
                if isinstance(task, list):
                    self.batch(*task)
                if isinstance(task, dict):
                    self.batch(**task)
            return self

        def wait(self, duration:float):
            """Blocking waiting for some duration (in seconds).

            Args:
                duration (float): The waiting time in seconds.

            Returns:
                Self: used for chain call.
            """
            self.pipeline.append(duration)
            return self

        def fetch(self):
            """Fetch the batch execution results.

            Args:
                None.

            Returns:
                list: The results in list in the order of batching enqueue sequence.
            """
            self.pipeline.append('fetch')
            return self

        def apply(self):
            """Apply the batch execution previously defined.

            Args:
                None.

            Returns:
                outputs (list): The outputs following the enqueue order of the batched tasks.
            """
            while self.pipeline:
                item = self.pipeline[0] #view
                if isinstance(item, tuple):                     ## * --> tasks
                    self.task_list.append(item)
                else:
                    if self.task_list: self._apply_tasks()      ## task --> tid
                    if isinstance(item, str) and item=='fetch': ## tid --> outputs
                        self._apply_fetch()
                    elif isinstance(item, float) or isinstance(item, int): ## (await)
                        time.sleep(item)
                self.pipeline.pop(0) #pop
            ##
            outputs = self._apply_outputs()
            return outputs

        pass

    def __init__(self, client:str='', addr:str='127.0.0.1', port:int=0):
        self.client = client
        self.executor = Connector.BatchExecutor(self)
        addr = addr if addr else ''
        port = port if port else IPC_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((addr, port))
        pass

    def list_all(self) -> dict:
        """List all online clients."""
        return self.handle('list_all', {})

    def describe(self) -> dict:
        """Return the available functions on the connected client."""
        return self.handle('describe', {})

    def reload(self) -> dict:
        """Request remote manifest to reload.

        Returns:
            dict: The response information.
        """
        return self.handle('reload', {})

    def info(self, function:str) -> dict:
        """Return the details of the function on the connected client.

        Args:
            function (str): The function name.

        Returns:
            dict: The dictionary object contains full manifest of the function.
        """
        return self.handle('info', {'function':function})

    def sync_code(self, basename:str):
        """Push the codebase on server to the client.

        Args:
            basename (str): The codebase name.

        Returns:
            dict: The response information.
        """
        return self.handle('sync_code', {'basename':basename})

    def execute(self, function:str, parameters:dict={}, timeout:float=-1) -> str:
        """Execute the function asynchronously, return instantly with task id.

        Args:
            function (str): The function name.
            parameters (dict): The parameters provided for the function. The absent values will use the default values in the manifest.
            timeout (float): The longest time in seconds waiting for the outputs from function execution.

        Returns:
            str: The task ID.
        """
        args = { 'function':function, 'parameters':parameters, 'timeout':timeout }
        res = self.handle('execute', args)
        return res['tid']

    def fetch(self, tid:str) -> dict:
        """Fetch the previous function execution results with task id.

        Args:
            tid (str): Task ID obtained from `Connector.execute`.

        Returns:
            dict: the output collected in dictionary struct.
        """
        return self.handle('fetch', {'tid':tid})

    def batch(self, *args, **kwargs):
        """Batch execution by simultaneously sending commands, then use `.apply` to apply send action.

        Args:
            client (str): The client name (or the server name).
            function (str): The function names.
            parameters (dict): The parameters provided for the function. The absent values will use the default values in the manifest.
            timeout (float): The longest time in seconds waiting for the outputs from function execution.

        Returns:
            Self: used for chain call.
        """
        return self.executor.batch(*args, **kwargs)

    def batch_all(self, *args, **kwargs):
        """Batch all the executions in list format.

        Args:
            task_list (list): the executions in list format.

        Returns:
            Self: used for chain call.
        """
        return self.executor.batch_all(*args, **kwargs)

    def apply(self):
        """Apply the batch execution previously defined.

        Args:
            None.

        Returns:
            outputs (list): The outputs following the enqueue order of the batched tasks.
        """
        return self.executor.apply()

    pass

def master_main(args):
    try:
        manifest = open('./manifest.json')
    except:
        manifest = {}
    else:
        manifest = json.load( manifest )
    master = MasterDaemon(args.port, args.ipc_port, manifest=manifest)
    master.start()
    pass

def slave_main(args):
    manifest = open('./manifest.json')
    manifest = json.load( manifest )
    ##
    slave = SlaveDaemon(args.port, manifest, args.client, alt_name=args.name)
    slave.start()
    pass

def main():
    parser = argparse.ArgumentParser(description='All-in-one cluster control tap.')
    parser.add_argument('-p', '--port', type=int, nargs='?', default=SERVER_PORT, help='(Optional) server port.')
    ##
    s_group = parser.add_argument_group('Server specific')
    s_group.add_argument('-s', '--server', action='store_true', help='run in server mode.')
    s_group.add_argument('--ipc-port', type=int, nargs='?', default=IPC_PORT, help='(Optional) external IPC port.')
    ##
    c_group = parser.add_argument_group('Client specific')
    c_group.add_argument('-c', '--client', type=str, default='', nargs='?', help='run in client mode.')
    c_group.add_argument('-n', '--name', type=str, default='', nargs='?', help='(Optional) specify custom client name.')
    ##
    args = parser.parse_args()
    if args.client or args.client==None:
        slave_main(args)
    elif args.server:
        master_main(args)
    else:
        print('Please specify client mode or server mode.')
    pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e #pass
    finally:
        pass #exit()
