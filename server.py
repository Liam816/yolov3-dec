import socket

def udp_server(host='0.0.0.0', port=12345):
    # 创建一个UDP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 绑定到指定的地址和端口
    server_socket.bind((host, port))
    
    print(f"UDP服务器已启动，正在监听 {host}:{port}")
    
    while True:
        # 接收来自客户端的消息
        message, client_address = server_socket.recvfrom(4096)
        print(f"收到来自 {client_address} 的消息: {message.decode()}")
        
        # 发送响应消息给客户端
        response_message = f"已收到你的消息: {message.decode()}"
        server_socket.sendto(response_message.encode(), client_address)

if __name__ == "__main__":
    udp_server()
