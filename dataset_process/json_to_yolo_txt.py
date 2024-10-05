import json
import os


# # label_map_reverse = {'worker': '0', 'robot_body': '1', 'robot_base': '2', 'robot_forearm': '3'}
# # label_map_reverse = {'human': '0', 'robot': '1', 'end': '2'}
# label_map_reverse = {'robot': '0', 'trunk': '1', 'head': '2'}


# 读取labelme标注文件
def read_labelme_annotation(labelme_annotation_path):
    with open(labelme_annotation_path, 'r', encoding='utf-8') as f:
        labelme_dict = json.load(f)
    return labelme_dict


# labelme标注文件转yolo格式
def labelme_anno2yolo(label_map, labelme_dict, img_path):
    yolo_anno_list = []
    img_wh = [labelme_dict['imageWidth'], labelme_dict['imageHeight']]
    for shape in labelme_dict['shapes']:
        label_name = shape['label']
        # label_index = label_map_reverse[label_name]
        label_index = label_map[label_name]
        x_min = shape['points'][0][0]
        y_min = shape['points'][0][1]
        x_max = shape['points'][1][0]
        y_max = shape['points'][1][1]

        x_center = (x_min + x_max) / 2 / img_wh[0]
        y_center = (y_min + y_max) / 2 / img_wh[1]
        width = (x_max - x_min) / img_wh[0]
        height = (y_max - y_min) / img_wh[1]

        yolo_anno_list.append("{} {} {} {} {}".format(label_index, x_center, y_center, width, height))
    return yolo_anno_list


def save_yolo_annotation(yolo_anno_list, save_path):
    with open(save_path, 'w') as f:
        for yolo_anno in yolo_anno_list:
            f.write(yolo_anno)
            f.write("\n")


def batch_labelme2yolo(label_map, labelme_annotation_dir, yolo_annotation_save_dir):
    labelme_annotation_list = os.listdir(labelme_annotation_dir)
    for labelme_annotation in labelme_annotation_list:
        labelme_annotation_path = os.path.join(labelme_annotation_dir, labelme_annotation)
        img_path = labelme_annotation_path.split(".")[0] + ".jpg"
        labelme_dict = read_labelme_annotation(labelme_annotation_path)
        yolo_anno_list = labelme_anno2yolo(label_map, labelme_dict, img_path)
        yolo_annotation_save_path = os.path.join(yolo_annotation_save_dir, labelme_annotation.split(".")[0] + ".txt")
        save_yolo_annotation(yolo_anno_list, yolo_annotation_save_path)


if __name__ == '__main__':
    # label_map_reverse = {'worker': '0', 'robot_body': '1', 'robot_base': '2', 'robot_forearm': '3'}
    # label_map_reverse = {'human': '0', 'robot': '1', 'end': '2'}
    label_map_reverse = {'robot': '0', 'trunk': '1', 'head': '2'}

    # labelme_annotation_dir = r"C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_worker\labels_json"
    # yolo_annotation_save_dir = r"C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_worker\labels_yolo_txt"

    # labelme_annotation_dir = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\labels_json"
    # yolo_annotation_save_dir = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\labels_yolo_txt"

    labelme_annotation_dir = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\labels_json"
    yolo_annotation_save_dir = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\labels_yolo_txt"

    batch_labelme2yolo(label_map_reverse, labelme_annotation_dir, yolo_annotation_save_dir)




