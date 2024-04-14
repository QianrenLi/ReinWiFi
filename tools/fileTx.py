
import socket 
import argparse
import time
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-i', '--ip', nargs='?')
    parser.add_argument ('-p', '--port', nargs='?', type=int, default=15555)
    parser.add_argument ('-f', '--file', nargs='+')
    return parser

# Creating Client Socket 
if __name__ == '__main__': 

    parser = createParser()
    namespace = parser.parse_args()
    filenames = namespace.file
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# Connecting with Server 
    host = namespace.ip
    port = namespace.port
    sock.connect((host, port)) 
    for filename in filenames:
        try: 
           # Reading file and sending data to server 
            fi = open(filename, "r") 
            data = fi.read() 
            if not data: 
                continue
            else:
                sock.send((str(filename) + "\n").encode())
            while data: 
                sock.send(str(data).encode()) 
                data = fi.read() 
            # File is closed after data is sent 
            fi.close()
            # send file end signal
            # sock.send("@@EndFile@@".encode())
            print("Tx end")
  
        except IOError: 
            print('You entered an invalid filename! Please enter a valid name') 
    ## close connection
    sock.close()