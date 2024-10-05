# 指定一个yolo标注与一个图片路径，然后运行yolo_label_vis.py，显示出图片和标注框

import cv2
import os

# img_path = "F:/yolo_v8_pcb_defect/datasets/pcb_real_defect_test_xiuzheng/VOCdevkit/JPEGImages/000000000000.jpg"
# yolo_annotation_path = "F:/yolo_v8_pcb_defect/datasets/pcb_real_defect_test_xiuzheng/VOCdevkit/txt/000000000000.txt"

img_path = "./JPEGImages/00001.bmp"
yolo_annotation_path = "./txt/00001.txt"

img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
with open(yolo_annotation_path, 'r') as f:
    lines = f.readlines()
lines_strip = [line.strip("\n") for line in lines]

for line in lines_strip:
    label_name = line.split(" ")[0]
    x_center = float(line.split(" ")[1])
    y_center = float(line.split(" ")[2])
    width = float(line.split(" ")[3])
    height = float(line.split(" ")[4])

    x_min = int((x_center - width / 2) * 640)
    y_min = int((y_center - height / 2) * 640)
    x_max = int((x_center + width / 2) * 640)
    y_max = int((y_center + height / 2) * 640)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(img, label_name, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
