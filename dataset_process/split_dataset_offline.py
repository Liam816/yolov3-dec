import os
import random
import math
import shutil


def main(img_root, label_root, dst_root, train_ratio, test_ratio):
    temp1 = os.path.join(dst_root, 'images')
    temp2 = os.path.join(dst_root, 'labels')
    temp1_1 = os.path.join(temp1, 'train')
    temp1_2 = os.path.join(temp1, 'val')
    temp1_3 = os.path.join(temp1, 'test')
    temp2_1 = os.path.join(temp2, 'train')
    temp2_2 = os.path.join(temp2, 'val')
    temp2_3 = os.path.join(temp2, 'test')

    dirs_to_create = list()
    dirs_to_create.append(dst_root)
    dirs_to_create.append(temp1)
    dirs_to_create.append(temp2)
    dirs_to_create.append(temp1_1)
    dirs_to_create.append(temp1_2)
    dirs_to_create.append(temp1_3)
    dirs_to_create.append(temp2_1)
    dirs_to_create.append(temp2_2)
    dirs_to_create.append(temp2_3)

    for dir_path in dirs_to_create:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    # if not os.path.isdir(temp1):
    #     os.mkdir(temp1)
    # if not os.path.isdir(temp2):
    #     os.mkdir(temp2)
    #
    # if not os.path.isdir(temp1):
    #     os.mkdir(temp1)
    # if not os.path.isdir(temp2):
    #     os.mkdir(temp2)
    #
    # if not os.path.isdir(temp1):
    #     os.mkdir(temp1)
    # if not os.path.isdir(temp2):
    #     os.mkdir(temp2)
    #
    # if not os.path.isdir(temp1):
    #     os.mkdir(temp1)
    # if not os.path.isdir(temp2):
    #     os.mkdir(temp2)

    img_name_list = os.listdir(img_root)
    label_name_list = os.listdir(label_root)
    sample_nums = len(img_name_list)
    assert len(img_name_list) == len(label_name_list), print("The amounts of images and labels are not same")

    index_list = list(range(sample_nums))
    # print("index_list:\n", index_list)

    random.seed(816)
    random.shuffle(index_list)

    train_nums = math.floor(train_ratio * sample_nums)
    test_nums = math.floor(test_ratio * sample_nums)
    val_nums = sample_nums - train_nums - test_nums

    print("train:{} val:{} test:{}".format(train_nums, val_nums, test_nums))
    print("0 : {}".format(train_nums))
    print("{} : {}".format(train_nums, train_nums + val_nums))
    print("{} : {}".format(train_nums + val_nums, train_nums + val_nums + test_nums))

    train_index_list = index_list[0: train_nums]
    val_index_list = index_list[train_nums: train_nums + val_nums]
    test_index_list = index_list[train_nums + val_nums: train_nums + val_nums + test_nums]

    # train
    for idx in train_index_list:
        img = img_name_list[idx]
        label = label_name_list[idx]

        img_src_path = os.path.join(img_root, img)
        label_src_path = os.path.join(label_root, label)

        img_dst_path = os.path.join(dst_root, 'images', 'train', img)
        label_dst_path = os.path.join(dst_root, 'labels', 'train', label)

        # print("img_src_path:{}".format(img_src_path))
        # print("label_src_path:{}".format(label_src_path))
        # print("img_dst_path:{}".format(img_dst_path))
        # print("label_dst_path:{}".format(label_dst_path))

        shutil.copy(img_src_path, img_dst_path)
        shutil.copy(label_src_path, label_dst_path)

    for idx in val_index_list:
        img = img_name_list[idx]
        label = label_name_list[idx]

        img_src_path = os.path.join(img_root, img)
        label_src_path = os.path.join(label_root, label)

        img_dst_path = os.path.join(dst_root, 'images', 'val', img)
        label_dst_path = os.path.join(dst_root, 'labels', 'val', label)

        shutil.copy(img_src_path, img_dst_path)
        shutil.copy(label_src_path, label_dst_path)

    for idx in test_index_list:
        img = img_name_list[idx]
        label = label_name_list[idx]

        img_src_path = os.path.join(img_root, img)
        label_src_path = os.path.join(label_root, label)

        img_dst_path = os.path.join(dst_root, 'images', 'test', img)
        label_dst_path = os.path.join(dst_root, 'labels', 'test', label)

        shutil.copy(img_src_path, img_dst_path)
        shutil.copy(label_src_path, label_dst_path)


if __name__ == "__main__":
    # img_root = r"C:\Users\ping.he\Desktop\liam\dataset\labelme\img_data"
    # label_root = r"C:\Users\ping.he\Desktop\liam\dataset\labelme\label_yolo_txt"
    # dst_root = r"C:\Users\ping.he\Desktop\liam\dataset\pedestrians"
    # train_ratio = 0.625
    # test_ratio = 0.25

    # img_root = r"C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_worker\images_data"
    # label_root = r"C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_worker\labels_yolo_txt"
    # dst_root = r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_worker\weld_robot_and_worker'
    # # NOTE: val = train - test
    # train_ratio = 0.7
    # test_ratio = 0.2

    # img_root = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\images_data"
    # label_root = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\labels_yolo_txt"
    # dst_root = r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\FSIE_lab_scenario'
    # # NOTE: val = train - test
    # train_ratio = 0.8
    # test_ratio = 0.1

    img_root = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\images_data_50_1"
    label_root = r"C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\labels_yolo_txt"
    dst_root = r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\FSIE_lab_bev_scenario'
    # NOTE: val = train - test
    train_ratio = 0.9
    test_ratio = 0.0

    main(img_root, label_root, dst_root, train_ratio, test_ratio)

    # temp1 = os.path.join(dst_root, 'images')
    # if not os.path.isdir(temp1):
    #     os.mkdir(temp1)




