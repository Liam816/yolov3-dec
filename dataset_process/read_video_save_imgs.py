import cv2
import time
import os
import shutil


class Worker:
    def __init__(self, video_path=None, frame_images_root=None):
        pass
        # self.video_path = video_path
        # self.video_file_name = os.path.basename(video_path).split('.')[0]
        # self.frame_images_root = frame_images_root

    def read_video_save_imgs(self, video_path, frame_images_root=None, start_id=0, frames_nums=None, saving=True):
        # # NOTE: 如果没有指定采样帧图片保存的路径 就在视频文件的目录下新建一个frame_images目录
        # if self.frame_images_root is None:
        #     self.frame_images_root = os.path.join(os.path.dirname(video_src_path), 'frame_images')
        # if not os.path.exists(self.frame_images_root):
        #     os.mkdir(self.frame_images_root)
        # dst_dir = os.path.join(self.frame_images_root, self.video_file_name)
        #
        # cap = cv2.VideoCapture(self.video_path)  # 导入视频，可以将视频放入和程序所在的同一目录下，也可以放置别的目录，修改对应的路径即可，我所用的是将视频文件放置当前目录下的情况。
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("video '{}' FPS:{}".format(self.video_file_name, fps))

        video_file_name = os.path.basename(video_path).split('.')[0]
        # NOTE: 如果没有指定采样帧图片保存的路径 就在视频文件的目录下新建一个frame_images目录
        if frame_images_root is None:
            frame_images_root = os.path.join(os.path.dirname(video_src_path), 'frame_images')
        if not os.path.exists(frame_images_root):
            os.mkdir(frame_images_root)
        dst_dir = os.path.join(frame_images_root, video_file_name)
        print('dst_dir:{}'.format(dst_dir))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        cap = cv2.VideoCapture(video_path)  # 导入视频，可以将视频放入和程序所在的同一目录下，也可以放置别的目录，修改对应的路径即可，我所用的是将视频文件放置当前目录下的情况。
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("video '{}' FPS:{}".format(video_file_name, fps))

        frame_id = start_id
        while True:
            _, im = cap.read()
            if im is None:
                break
            cv2.imshow('name', im)

            if saving:
                file_name = dst_dir + "/" + str(frame_id) + ".jpg"
                cv2.imwrite(file_name, im)
                frame_id += 1  # 间隔1帧取一帧

            key = cv2.waitKey(2) & 0xFF
            if (key == ord('q')) | (key == 27):
                break

            if frames_nums and frame_id - start_id >= frames_nums:
                break

        return frame_id

    def sample_imgs_for_dataset(self, img_root, dst_root, start, end, interval):
        """
        :param img_root:
        :param dst_root:
        :param start: 百分比的起点
        :param end: 百分比的终点 例如10%-90%
        :param interval: 间隔n帧采样一次
        :return:
        """
        # 获取目录下所有文件的路径
        file_names = os.listdir(img_root)
        file_nums = len(file_names)
        file_names = file_names[int(file_nums * start): int(file_nums * end)]
        file_names = file_names[::interval]

        if not os.path.exists(dst_root):
            os.mkdir(dst_root)

        file_src_paths = [os.path.join(img_root, f) for f in file_names]
        file_dst_paths = [os.path.join(dst_root, f) for f in file_names]

        for i in range(len(file_src_paths)):
            shutil.copy(file_src_paths[i], file_dst_paths[i])


def main(video_path, images_root, start_id):
    cap = cv2.VideoCapture(video_path)  # 导入视频，可以将视频放入和程序所在的同一目录下，也可以放置别的目录，修改对应的路径即可，我所用的是将视频文件放置当前目录下的情况。
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:{}".format(fps))
    # exit()

    saving = True  # 控制视频是否将视频逐帧保存为图片
    frame_id = 0

    if not os.path.exists(images_root):
        os.mkdir(images_root)

    dir1 = images_root
    # dir1 = dir1 + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # os.mkdir(dir1)

    saving = True
    frame_id = start_id
    while True:
        _, im = cap.read()
        if im is None:
            break
        cv2.imshow('name', im)
        key = cv2.waitKey(10) & 0xFF
        if saving:
            file_name = dir1 + "/" + str(frame_id)
            cv2.imwrite(file_name + ".jpg", im)
            frame_id += 1

        # if frame_id >= 200:  # fps=10 20s
        #     break

        if (key == ord('q')) | (key == 27):
            break

        # 如果需要按键控制开始视频保存为图像的时机，可以使用下面的代码。
        # if key == ord('s') or key == ord('S'):
        #     if not saving:
        #         dir1 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        #         os.mkdir(dir1)
        #         saving = True
        #         frame_id = 0
        #     else:
        #         saving = False
        #
        # if saving:
        #     file_name = dir1 + "\\" + str(frame_id)
        #     cv2.imwrite(file_name + ".jpg", im)
        #
        #     frame_id += 1
        # if (key == ord('q')) | (key == 27):
        #     break


def analysis_video(src_video_path):
    cap = cv2.VideoCapture(src_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frames:{}  |  FPS:{}'.format(frame_count, fps))
    print(f'Resolution: {height} x {width}')


def get_one_frame(src_video_path):
    cap = cv2.VideoCapture(src_video_path)
    # 获取视频帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算中间帧的索引
    middle_frame_index = frame_count // 2
    # 设置视频帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    # 读取中间帧
    ret, frame = cap.read()

    if ret:
        # 展示图像（如果需要）
        cv2.imshow('Middle Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存图像
        cv2.imwrite('middle_frame.png', frame)
    else:
        print("Error: Could not read frame.")

    # 释放视频捕获对象
    cap.release()


def crop_video(src_video_path):
    cap = cv2.VideoCapture(src_video_path)

    # 获取视频帧率和帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 根据帧率计算视频的总时长，然后确定最后10秒视频的开始位置
    duration = frame_count / fps
    start_time = max(duration - 65, 0)  # 防止视频小于10秒
    start_frame = int(start_time * fps)

    # 设置视频帧位置到最后10秒的起始处
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = 'dst_video.mp4'
    # 定义输出视频的编码器和输出对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以更改为其他编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # 从视频中读取并写入新视频文件
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            # 检查是否已达到视频末尾
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_count:
                break
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    print(f"Successfully saved the last 10 seconds of the video to {output_path}")


if __name__ == "__main__":

    # video_src_path = r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\BV1Fb4y1C79W.mp4'
    # # imgs_dst_root = r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\images\BV1z24y197iR'
    # # main(video_src_path, imgs_dst_root, 1763)

    # # worker = Worker()
    # # last_frame_id = worker.read_video_save_imgs(video_src_path, None, 0)
    # # print('last_frame_id:{}'.format(last_frame_id))
    #
    # worker = Worker()
    # worker.sample_imgs_for_dataset(r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\frame_images\BV1Fb4y1C79W',
    #                                r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\images_data',
    #                                0.0, 0.7, 8)
    # worker.sample_imgs_for_dataset(r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\frame_images\BV1Sz4y1w7nA',
    #                                r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\images_data',
    #                                0.0, 0.99, 23)
    # worker.sample_imgs_for_dataset(r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\frame_images\BV1z24y197iR',
    #                                r'C:\Users\ping.he\Desktop\liam\dataset\weld_robot_and_human\images_data',
    #                                0.093, 0.648, 36)

    # get_one_frame(r'C:\Users\ping.he\OneDrive - zju.edu.cn\FSIE\东方电气\FSIE_detection_result.mp4')
    # crop_video(r'C:\Users\ping.he\OneDrive - zju.edu.cn\FSIE\东方电气\FSIE_detection_result.mp4')
    # exit()


    # video_src_path = r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\FSIE.mp4'
    video_src_path = r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_BEV_scenario\bev_scenario.mp4'
    analysis_video(video_src_path)
    exit()

    # worker = Worker()
    # # last_frame_id = worker.read_video_save_imgs(video_src_path, None, 0)
    # # print('last_frame_id:{}'.format(last_frame_id))
    #
    # worker.sample_imgs_for_dataset(r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\frame_images\FSIE',
    #                                r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_scenario\images_data',
    #                                0.0, 0.75, 37)

    video_src_path = r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\bev_scenario.mp4'
    worker = Worker()
    # last_frame_id = worker.read_video_save_imgs(video_src_path, None, 0)
    # print('last_frame_id:{}'.format(last_frame_id))
    worker.sample_imgs_for_dataset(r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\frame_images\bev_scenario',
                                   r'C:\Users\ping.he\Desktop\liam\dataset\FSIE_lab_bev_scenario\images_data',
                                   0.05, 0.75, 37)

