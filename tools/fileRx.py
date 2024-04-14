import socket 
import os

def argParser():
    import argparse
    parser = argparse.ArgumentParser(description='File Receiver')
    parser.add_argument('--host', default='192.168.3.82')
    parser.add_argument('--folder', default='./logs')
    parser.add_argument('--port', default=15555)
    args = parser.parse_args()
    return args

def receiver(host, port, folder):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    host = host
    port = port

    totalclient = 1
    sock.bind((host, port)) 
    sock.listen(totalclient) 
    # Establishing Connections 
    connections = [] 
    print('Initiating clients') 
  
    fileno = 0
    idx = 0
    if not os.path.exists(folder):
        os.makedirs(folder)
    while True:
        conn, addr = sock.accept() 
        while True:
            # Receiving File Data 
            idx += 1
            data = conn.recv(1024).decode() 
            if not data: 
                break
            # Creating a new file at server end and writing the data 
            filename = f'{folder}/output'+str(fileno)+'.txt'
            ## if file exists, update fileno
            while os.path.exists(filename):
                fileno = fileno+1
                filename = f'{folder}/output'+str(fileno)+'.txt'  
            print(filename)
            fo = open(filename, "w")
            ## write current time
            import datetime
            now = datetime.datetime.now()
            fo.write(now.strftime("%Y-%m-%d %H:%M:%S\n"))
            while data: 
                if not data: 
                    break
                else: 
                    # Closing the file if end of file reached
                    if "@@EndFile@@" in data:
                        data = data.replace("@@EndFile@@", "")
                        fo.write(data)
                        break
                    fo.write(data) 
                    data = conn.recv(1024).decode() 

            fo.write("@@EndFile@@\n")
            print('Receiving file from client', idx) 
            print('Received successfully! New filename is:', filename) 
            fo.close()
            break
        break


if __name__ == '__main__': 
    args = argParser()
    # Defining Socket 
    receiver(args.host, args.port, args.folder)