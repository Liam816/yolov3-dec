

## YOLO数据集制作流程（从原始视频开始）
1. 运行read_video_save_imgs.py，整个视频抽帧得到的图片会保存在frame_images文件夹中，选取其中要作为数据集的图片放到images_data文件夹中；
2. 先把标注好的json文件统一放到labels_json文件夹中，再创建一个labels_yolo_txt文件夹用来存在标签的txt文件，运行json_to_yolo_txt.py；
3. split_dataset_offline.py
