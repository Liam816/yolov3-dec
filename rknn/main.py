import cv2
import time
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc
import sys
import wiringpi
print(wiringpi.__file__)
from wiringpi import GPIO
import subprocess

'''
terminal command:
gpio -v
gpio readall
'''

# res = subprocess.run(['gpio', 'mode', '2', 'out'], capture_output=True, text=True)
# print(res.stdout)
# res = subprocess.run(['gpio', 'write', '2', '1'], capture_output=True, text=True)
# time.sleep(2)
# res = subprocess.run(['gpio', 'write', '2', '0'], capture_output=True, text=True)
# res = subprocess.run(['gpio', 'mode', '2', 'in'], capture_output=True, text=True)
# exit()

def gpio_test():
    print('hello')
    wiringpi.wiringPiSetup()
    print('hello')
    gpio_num = wiringpi.getGpioNum()
    print('hello')
    print(gpio_num)
    exit()
    for i in range(0, gpio_num):
        wiringpi.pinMode(i, GPIO.OUTPUT)



if __name__ == '__main__':

    # gpio_test()
    # exit()

    video_path = '/home/orangepi/liam/projects/yolov3-dec/dataset_process/dec_real_scenario.mp4'
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    modelPath = '/home/orangepi/liam/projects/weights/DEC_yolov5n/dec_yolov5n_best.rknn'
    # 线程数
    TPEs = 6
    # 初始化rknn池
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        TPEs=TPEs,
        func=myFunc)
    
    # 初始化异步所需要的帧
    if (cap.isOpened()):
        for i in range(TPEs + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(frame)
    
    frames, loopTime, initTime = 0, time.time(), time.time()
    while (cap.isOpened()):
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, flag = pool.get()
        if flag == False:
            break
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
            loopTime = time.time()
    
    print("总平均帧率\t", frames / (time.time() - initTime))
    # 释放cap和rknn线程池
    cap.release()
    cv2.destroyAllWindows()
    pool.release()