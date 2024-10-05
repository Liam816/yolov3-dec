# # server.py (运行在Jetson Nano上)
# import socket

# # # 选择一个IP地址和端口号
# # server_ip = '10.8.118.122' # 这里使用Jetson Nano的本地网络IP地址
# # server_port = 10086

# # # # 创建socket对象
# # # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # # # 绑定到指定的IP地址和端口上
# # # server_socket.bind((server_ip, server_port))

# # # # 开始监听
# # # server_socket.listen()

# # # print(f"Listening on {server_ip}:{server_port}")

# # # # 接受连接
# # # connection, address = server_socket.accept()
# # # print(f"Connected to {address}")

# # # # 发送数据
# # # data_to_send = "Hello from Jetson Nano!"
# # # connection.sendall(data_to_send.encode())

# # # # 关闭连接
# # # connection.close()

# # try:
# #     # 创建socket对象
# #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# #     # 绑定到指定的IP地址和端口上
# #     server_socket.bind((server_ip, server_port))

# #     # 开始监听
# #     server_socket.listen()
# #     print(f"Listening on {server_ip}:{server_port}")

# #     # 接受连接
# #     connection, address = server_socket.accept()
# #     print(f"Connected to {address}")

# #     # 发送数据
# #     data_to_send = "Hello from Jetson Nano!"
# #     connection.sendall(data_to_send.encode())

# #     # 关闭连接
# #     connection.close()

# # except KeyboardInterrupt:
# #     print("Server is terminating...")
# #     server_socket.close()


# if __name__ == '__main__':
 
#     # 创建 socket 对象
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
#     # 获取本地主机名
#     host = socket.gethostname()
#     print('host:', host)

#     # 设置端口号
#     port = 12345
    
#     # 绑定端口
#     server_socket.bind((host, port))
    
#     # 设置最大连接数，超过后排队
#     server_socket.listen(5)
    
#     while True:
#         # 建立客户端连接
#         client_socket, addr = server_socket.accept()
    
#         print(f"连接地址: {addr}")
    
#         message = '服务器响应！' + "\r\n"
#         client_socket.send(message.encode('ascii'))
    
#         client_socket.close()


# -*- coding:utf-8 -*-
# @Author: 喵酱
# @time: 2024 - 04 -10
# @File: server.py
# desc:
import socket
 
def main():
    # 创建套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定IP地址和端口
    server_socket.bind(('localhost', 8888))
    # 监听连接
    server_socket.listen(1)
 
    print("服务启动，等待客户端连接... ")
 
    # 接受客户端连接
    client_socket, client_address = server_socket.accept()
    print(f"客户端 {client_address} 已连接")
 
    while True:
        try:
            # 接收客户端消息
            message = client_socket.recv(1024).decode()
            print("A:", message)
 
            # 发送消息给客户端
            message_b = input("输入消息：")
            client_socket.send(message_b.encode())
        except Exception as e:
            print("发生异常:", e)
            break
 
    # 关闭连接
    client_socket.close()
    server_socket.close()


def test2():
    
    # 创建套接字
    tcp_server = socket(AF_INET, SOCK_STREAM)
    # 将192.168.1.2换为本机的IP，8000为使用的接口（保持8000即可，可根据需要修改）
    address = ("10.8.118.122", 8000)
    tcp_server.bind(address)
    # 启动被动连接，设置多少个客户端可以连接
    tcp_server.listen(128)
    # 使用socket创建的套接字默认的属性是主动的
    # 使用listen将其变为被动的，这样就可以接收别人的链接了

    # 创建接收
    # 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
    client_socket, clientAddr = tcp_server.accept()
    # client_socket用来为这个客户端服务，相当于的tcp_server套接字的代理
    # tcp_server_socket就可以省下来专门等待其他新客户端的链接
    # 这里clientAddr存放的就是连接服务器的客户端地址

    from_client_msg=client_socket.recv(1024)
    while(1):
        #接收对方发送过来的数据
        from_client_msg = client_socket.recv(1024) #接收1024给字节,这里recv接收的不再是元组，区别UDP
        if(from_client_msg=="exit"): # 断开连接
            break
        print("接收的数据：",from_client_msg.decode("gbk"))
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #发送数据给客户端
        send_data = client_socket.send((str(now_time)+" 已收到！").encode("gbk"))
    client_socket.close()


def test3():
    # udp_server.py
    import socket

    # 选择一个IP地址和端口号
    server_ip = '192.168.1.89'  # 这里使用Jetson Nano的本地网络IP地址
    server_port = 10086

    # 创建socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定到指定的IP地址和端口上
    server_socket.bind((server_ip, server_port))

    print(f"UDP server up and listening on {server_ip}:{server_port}")

    # 等待客户端发送数据
    while True:
        data, address = server_socket.recvfrom(1024)  # 缓冲区大小设置为1024字节
        print(f"Received message from {address}: {data.decode()}")

        # 发送响应数据
        response = "Hello from Jetson Nano!"
        server_socket.sendto(response.encode(), address)


def test4():





if __name__ == "__main__":
    # main()
    # test2()
    # test3()
    test4()

    


