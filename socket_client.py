import socket

# UDP IP地址和端口号
UDP_IP = "0.0.0.0"  # 监听所有进入的地址
UDP_PORT = 5005

# 创建socket
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
sock.bind((UDP_IP, UDP_PORT))

print("Server is waiting for messages...")

try:
    while True:
        data, addr = sock.recvfrom(1024)  # 缓冲大小为1024字节
        print(f"Received message: {data} from {addr}")

except KeyboardInterrupt:
    print("Server has been stopped.")
finally:
    sock.close()
