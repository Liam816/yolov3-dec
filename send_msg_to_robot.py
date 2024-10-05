import socket
import time


def udp_client(server_host='192.168.10.89', server_port=10040, message=None):
    # 创建一个UDP套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 要发送的消息
    # message = "Hello, UDP Server!"
    
    try:
        # encoded_msg = ""
        # for num in message:
        #     msg_byte = num.to_bytes(1, byteorder='big')
        #     # print(msg_byte)
        #     msg_byte_str = str(msg_byte)
        #     print(msg_byte_str)
        #     encoded_msg += msg_byte_str
        # print('encoded_msg:', encoded_msg)
        # # 发送消息给服务器
        # client_socket.sendto(encoded_msg, (server_host, server_port))

        client_socket.sendto(message, (server_host, server_port))
        
        print(f"message sent to {server_host}:{server_port}")

        # # 接收来自服务器的响应消息
        # response, server_address = client_socket.recvfrom(4096)
        # print(f"收到来自服务器的响应: {response.decode()}")
    
    finally:
        # 关闭套接字
        client_socket.close()


if __name__ == '__main__':
    # stop
    pause_message = "[59][45][52][43][20][00][04][00][03][01][00][01][00][00][00][00][39][39][39][39][39][39][39][39][83][00][01][00][01][10][00][00][01][00][00][00]"
    # cancel stop
    cancel_pause_message = "[59][45][52][43][20][00][04][00][03][01][00][01][00][00][00][00][39][39][39][39][39][39][39][39][83][00][01][00][01][10][00][00][02][00][00][00]"

    trimmed_message = pause_message.strip("[]")
    split_numbers = trimmed_message.split("][")
    print('split_numbers:\n', split_numbers)
    bytes_data = bytes.fromhex(''.join(split_numbers))
    print('bytes_data:\n', bytes_data)
    udp_client(message=bytes_data)
    
    time.sleep(2)

    trimmed_message = cancel_pause_message.strip("[]")
    split_numbers = trimmed_message.split("][")
    print('split_numbers:\n', split_numbers)
    bytes_data = bytes.fromhex(''.join(split_numbers))
    print('bytes_data:\n', bytes_data)
    udp_client(message=bytes_data)
    




    