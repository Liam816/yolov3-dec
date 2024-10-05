import cv2
import numpy as np
import multiprocessing as mp

def capture_from_camera(camera_id, queue):
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"无法从摄像头 {camera_id} 读取画面")
            break
        # cv2.imshow("frame", frame)
        queue.put(frame)
    cap.release()

def display_frames(queue1, queue2):
    while True:
        if not queue1.empty() and not queue2.empty():
            frame1 = queue1.get()
            frame2 = queue2.get()

            # 拼接两个画面
            frame1 = cv2.resize(frame1, (frame1.shape[1] // 2, frame1.shape[0] // 2))
            frame2 = cv2.resize(frame2, (frame2.shape[1] // 2, frame2.shape[0] // 2))
            combined_frame = np.hstack((frame1, frame2))

            cv2.imshow("Combined Camera Feed", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def capture_and_display(camera_id):
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return
    
    while True:
        # 读取每一帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 显示画面
        cv2.imshow("Camera Feed", frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()



def check_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"camera {i} valid")
            cap.release()
        else:
            print(f"camera {i} invalid")


if __name__ == "__main__":
    
    # check_camera()
    # exit()
    # capture_and_display(2)  # 尝试使用摄像头 ID 0
    # exit()

    # 创建两个队列来存储从摄像头读取的画面
    queue1 = mp.Queue()
    queue2 = mp.Queue()

    # 启动两个进程来读取两个摄像头的画面
    p1 = mp.Process(target=capture_from_camera, args=(0, queue1))
    p2 = mp.Process(target=capture_from_camera, args=(2, queue2))

    p1.start()
    p2.start()

    # 启动一个进程来显示拼接后的画面
    p3 = mp.Process(target=display_frames, args=(queue1, queue2))
    p3.start()

    # 等待所有进程完成
    p1.join()
    p2.join()
    p3.join()