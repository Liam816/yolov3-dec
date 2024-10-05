import cv2
import time



if __name__ == '__main__':
    # 初始化摄像头
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义MP4编解码器
    out = cv2.VideoWriter('dataset.mp4', fourcc, 20.0, (640, 480))

    start_time = time.time()
    while True:
        ret, frame = cap.read()  # 读取当前帧
        if not ret:
            print("无法获取视频帧")
            break
        
        # print('frame type:', type(frame))

        out.write(frame)  # 写视频帧到文件
        cv2.imshow('Camera', frame)  # 实时展示视频画面

        # 如果按下“q”键，退出循环，注意要在摄像头画面里按q
        if cv2.waitKey(1) == ord('q'):
            break

        # if (time.time() - start_time) % 10 == 0:
        #     print('current time:{.2f}'.format(time.time() - start_time))

        # 检查是否已经录制了3分钟
        if time.time() - start_time >= 180:  # 记录3分钟
            break

    # 释放摄像头资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()



