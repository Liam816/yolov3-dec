# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

import math
import numpy as np

import time
import Jetson.GPIO as GPIO
import serial
import socket


# serial
# ser = serial.Serial('/dev/ttyUSB0', '115200')
# if not ser.is_open:
#     ser.open()
# ser.flush()


host='0.0.0.0'
port=12345
# 创建一个UDP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定到指定的地址和端口
server_socket.bind((host, port))
server_socket.settimeout(0.005)
print(f"UDP服务器已启动 正在监听 {host}:{port}")

# gpio
GPIO.setmode(GPIO.BOARD)
# for emergency stop
GPIO.setup(11, GPIO.OUT)  # 8
GPIO.setup(12, GPIO.OUT)  # 7
# for speed adjustment
GPIO.setup(13, GPIO.OUT)  # 6
GPIO.setup(15, GPIO.OUT)  # 5



def get_line_center_length(start, end):
    center = (int(((start[0] + end[0]) / 2)), int(((start[1] + end[1]) / 2)))
    length = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    return center, length


def rectangle_invation(rect1:[], rect2:[]):
    # 解构矩形坐标
    (x1_A, y1_A), (x2_A, y2_A) = rect1
    (x1_B, y1_B), (x2_B, y2_B) = rect2
    
    # 检查矩形A和B是否不重叠
    if x1_A >= x2_B or x2_A <= x1_B:  # 检查左右边界
        return False
    if y1_A <= y2_B or y2_A >= y1_B:  # 检查上下边界
        return False

    # 矩形重叠
    return True


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    print('nosave:{}'.format(nosave))
    # exit()
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    print('save_img:{}'.format(save_img))
    # save_img = True  # LIAM
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    print('save_dir:{}'.format(save_dir))

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        dataset2 = LoadStreams('2', img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    print('dataset.mode:{}'.format(dataset.mode))

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # for (path, im, im0s, vid_cap, s), (path_2, im_2, im0s_2, vid_cap_2, s_2) in zip(dataset, dataset2):
    for event1, event2 in zip(dataset, dataset2):

        # 接收来自客户端的消息
        try:
            message, client_address = server_socket.recvfrom(4096)
            print(f"收到来自 {client_address} 的消息: {message.decode()}")
            tof_data = message.decode()
            decoded_tof_data = tof_data.split(',')
            tof1, switch1 = int(decoded_tof_data[0]), int(decoded_tof_data[1])
            if switch1 == 1:
                # for emergency stop
                GPIO.output(11, GPIO.LOW)  # 8
                GPIO.output(12, GPIO.LOW)  # 7
                time.sleep(0.01)
                # for elem in decoded_info:
                #     print(elem)
            if tof1 < 150:
                # decrease the robot speed giant ratio
                GPIO.output(13, GPIO.LOW)
                time.sleep(0.01)
                # for elem in decoded_info:
                #     print(elem)
        except:
            # print('No data received by socket.')
            pass

        event_list = [event1, event2]
        res_img_list = []

        start_time = time.time()

        for event in event_list:
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            path, im, im0s, vid_cap, s = event[0], event[1], event[2], event[3], event[4]

            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                robot_center_points = []
                trunk_center_points = []
                robot_corners = []
                trunk_corners = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if cls.cpu().item() == 1:  # 'robot'
                            x_ = int((xyxy[0].cpu().item() + xyxy[2].cpu().item()) / 2)
                            y_ = int((xyxy[1].cpu().item() + xyxy[3].cpu().item()) / 2)
                            robot_center_points.append((x_, y_))
                            robot_corners.append([(xyxy[0].cpu().item(), xyxy[1].cpu().item()), (xyxy[2].cpu().item(), xyxy[3].cpu().item())])

                        if cls.cpu().item() == 0:  # 'trunk'
                            x_ = int((xyxy[0].cpu().item() + xyxy[2].cpu().item()) / 2)
                            y_ = int((xyxy[1].cpu().item() + xyxy[3].cpu().item()) / 2)
                            # center_points.append((cls, (x_, y_)))
                            trunk_center_points.append((x_, y_))
                            trunk_corners.append([(xyxy[0].cpu().item(), xyxy[1].cpu().item()), (xyxy[2].cpu().item(), xyxy[3].cpu().item())])

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            # print('label:', label)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # exit()

                # Stream results
                im0 = annotator.result()

                # LIAM
                shield_y_field = 160
                dis_threshold = 100
                center_dis_flag = False
                dis_list_per_loop = []
                for rb_ct in robot_center_points:
                    for trunk_ct in trunk_center_points:
                        str_center, line_length = get_line_center_length(rb_ct, trunk_ct)
                        dis_list_per_loop.append(((trunk_ct[0], trunk_ct[1]), line_length))
                        if line_length <= dis_threshold and trunk_ct[1] >= shield_y_field:  # xy 10 pixels
                            center_dis_flag = True
                            cv2.line(im0, rb_ct, trunk_ct, (255, 0, 128), 3)

                rectangle_invation_flag = False
                for rb_corner in robot_corners:
                    for trunk_corner in trunk_corners:
                        # if rectangle_invation(rb_corner, trunk_corner) and trunk_ct[1] >= shield_y_field:
                        if rectangle_invation(rb_corner, trunk_corner):
                            rectangle_invation_flag = True
                            break
                
                # if rectangle_invation_flag:
                #     print('invation !!!!!!!!!!!!!!!!!')
                # print('rectangle_invation_flag:', rectangle_invation_flag)

                if center_dis_flag or rectangle_invation_flag:
                    red_mask = np.zeros_like(im0)
                    red_mask[:, :] = [0, 0, 255]
                    alpha = 0.3  # 透明度
                    im0 = cv2.addWeighted(im0, 1 - alpha, red_mask, alpha, 0)
                    cv2.putText(im0, 'WARNING', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    # decrease the robot speed tiny ratio
                    GPIO.output(13, GPIO.LOW)  # 5
                    # GPIO.output(11, GPIO.LOW)  # 8
                    # GPIO.output(12, GPIO.LOW)  # 7
                    time.sleep(0.02)

                # end_time = time.time()
                # cost_time = end_time - start_time
                # fps = round(1 / cost_time, 2)
                # fps_2_draw = "FPS:" + str(fps)
                # cv2.putText(im0, fps_2_draw, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                res_img_list.append(im0)

                # view_img = True
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(1)  # 1 millisecond
                    # cv2.waitKey(0)  # 1 millisecond
                # exit()

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # LIAM
                        # print('im0 type:', type(im0))
                        vid_writer[i].write(im0)

                GPIO.output(11, GPIO.HIGH)  # 8
                GPIO.output(12, GPIO.HIGH)  # 7
                GPIO.output(13, GPIO.HIGH)  # 6
                GPIO.output(15, GPIO.HIGH)  # 5

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        end_time = time.time()
        cost_time = end_time - start_time
        fps = round(1 / cost_time, 2)
        fps_2_draw = "FPS:" + str(fps)
        LOGGER.info(f"=== {fps_2_draw} ===")

        frame_list = []
        for img_candi in res_img_list:
            frame = cv2.resize(img_candi, (img_candi.shape[1] // 2, img_candi.shape[0] // 2))
            frame_list.append(frame)
        
        combined_frame = np.hstack(frame_list)
        cv2.imshow("Combined Camera Feed", combined_frame)
        # cv2.waitKey(1)  # 1 millisecond

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'yolov3-tiny.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def trigger_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)
    for i in range(10):
        GPIO.output(11, GPIO.LOW)
        time.sleep(3)
        GPIO.output(11, GPIO.HIGH)
        time.sleep(3)


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    # trigger_gpio()
    # hardware_init()
    opt = parse_opt()
    main(opt)


