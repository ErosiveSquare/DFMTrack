# DFM-Track 运行脚本

import numpy as np
import cv2
import os
import pandas as pd
import torch
import multiprocessing
from pathlib import Path

from boxmot.trackers.dfmtrack.DFMtrack2 import DFMTrack

os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO


# 设置使用的GPU设备
# os.environ['CUDA_VISIBLE_DEVICES'] = str(1)  # 根据需要选择GPU，例如 '0', '1', '0,1'

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比(Intersection over Union)

    参数:
    box1, box2: [x_min, y_min, x_max, y_max]

    返回:
    float: 交并比值
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def box_label(image, box, label='', color=(167, 146, 11), txt_color=(255, 255, 255)):
    """在图像上绘制边界框和标签"""
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        # 字体大小和粗细
        font_scale = 0.5
        thickness = 1

        # 获取文本尺寸
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 确保标签背景不会超出图像边界
        outside = p1[1] - h - 3 >= 0
        p2_text_bg = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

        # 绘制标签背景
        cv2.rectangle(image, p1, p2_text_bg, color, -1, cv2.LINE_AA)

        # 绘制标签文本
        text_pos = (p1[0], p1[1] - 5 if outside else p1[1] + h + 5)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness, cv2.LINE_AA)


def process_video(input_video_path, output_video_dir, output_gallery_dir, output_excel_dir, save_video, device_id):
    """
    处理单个视频文件，进行目标检测和跟踪。
    """
    # 选择设备
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"处理视频: {input_video_path}, 使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_excel_dir, exist_ok=True)

    # 生成输出文件路径
    video_filename = os.path.basename(input_video_path)
    video_filename_without_ext = os.path.splitext(video_filename)[0]
    output_video_path = os.path.join(output_video_dir, f'dfm_out_{video_filename_without_ext}.mp4')
    output_excel_path = os.path.join(output_excel_dir, f'{video_filename_without_ext}.xlsx')
    output_gallery_dir_for_video = os.path.join(output_gallery_dir, f'{video_filename_without_ext}')
    os.makedirs(output_gallery_dir_for_video, exist_ok=True)

    try:
        # --- 1. 初始化模型 ---
        print("正在加载YOLOv8检测模型...")
        # 注意：YOLO模型会自动使用指定的device
        model = YOLO('weights/yolov8m.pt', task='detect')
        print("YOLOv8加载成功。")

        # --- 2. 初始化 DFM-Track 跟踪器 ---
        print("正在初始化 DFM-Track 跟踪器...")
        tracker = DFMTrack(
            model_weights=Path('weights/clip_zhz.pt'),  # ReID模型权重路径
            device=str(device),  # 必须是字符串格式，如 'cuda:0' 或 'cpu'
            fp16=False,  # 是否使用半精度

            # DFM-Track 特定参数
            original_feature_dim=1280,  # 原始ReID特征维度
            reduced_feature_dim=64,  # 降维后的特征维度
            ukf_q_std=0.01,  # UKF过程噪声
            ukf_r_std=0.1,  # UKF测量噪声

            # 关联和生命周期参数
            mahalanobis_gate_thresh=0.95,  # 马氏距离门控
            cosine_dist_thresh=0.4,  # 余弦距离阈值
            det_thresh=0.3,  # 检测置信度阈值
            max_age=30,  # 最大丢失帧数
            min_hits=3  # 最小命中次数
        )
        print("DFM-Track 初始化成功。")

        # --- 3. 视频读写设置 ---
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息 - FPS: {fps}, 尺寸: {width}x{height}")

        videoWriter = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器更兼容
            videoWriter = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # --- 4. 逐帧处理 ---
        image_info_list = []
        frame_counter = 0
        continuous_id_counter = 0
        PERSON_CLASS_ID = 0  # 假设'person'类别ID为0
        track_history = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"视频读取结束: {input_video_path}")
                break

            frame_counter += 1
            if frame_counter % 10 == 0:
                print(f"正在处理第 {frame_counter} 帧...")

            # --- 目标检测 ---
            results = model(frame, conf=0.1, verbose=False)
            outputs = results[0].boxes.data.cpu().numpy()

            if outputs is not None and len(outputs) > 0:
                # 筛选'person'类别
                person_detections = outputs[outputs[:, 5] == PERSON_CLASS_ID]

                if len(person_detections) > 0:
                    # 准备给跟踪器的输入 [x1, y1, x2, y2, conf, cls]
                    dets_for_tracker = person_detections[:, :6]

                    # --- 目标跟踪 ---
                    tracks = tracker.update(dets_for_tracker, frame)

                    if len(tracks) > 0:
                        # DFM-Track输出格式: [x1, y1, x2, y2, id, conf, cls, det_idx]
                        for track in tracks:
                            bbox = track[:4]
                            track_id = int(track[4])

                            # ID重映射，使其从0开始连续
                            if track_id not in track_history:
                                track_history[track_id] = continuous_id_counter
                                continuous_track_id = continuous_id_counter
                                continuous_id_counter += 1
                            else:
                                continuous_track_id = track_history[track_id]

                            # 绘制边界框和ID
                            box_label(frame, bbox, f'ID: {continuous_track_id}')

                            # --- 保存裁剪图和信息 ---
                            x_min, y_min, x_max, y_max = map(int, bbox)
                            # 确保坐标在图像范围内
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(width, x_max)
                            y_max = min(height, y_max)

                            if x_max > x_min and y_max > y_min:
                                cropped_img = frame[y_min:y_max, x_min:x_max]
                                if cropped_img.size > 0:
                                    person_gallery_dir = os.path.join(output_gallery_dir_for_video,
                                                                      str(continuous_track_id))
                                    os.makedirs(person_gallery_dir, exist_ok=True)

                                    cropped_filename = os.path.join(person_gallery_dir, f'frame_{frame_counter}.jpg')
                                    cv2.imwrite(cropped_filename, cropped_img)

                                    # 归一化坐标
                                    norm_bbox = [
                                        round(x_min / width, 5),
                                        round(y_min / height, 5),
                                        round(x_max / width, 5),
                                        round(y_max / height, 5)
                                    ]
                                    image_info_list.append({
                                        'image_path': cropped_filename,
                                        'frame': frame_counter,
                                        'id': continuous_track_id,
                                        'BBox': norm_bbox
                                    })

            if save_video and videoWriter is not None:
                videoWriter.write(frame)

        # --- 5. 清理和保存 ---
        cap.release()
        if videoWriter is not None:
            videoWriter.release()
            print(f"输出视频已保存到: {output_video_path}")

        if image_info_list:
            df = pd.DataFrame(image_info_list)
            df.to_excel(output_excel_path, index=False)
            print(f"跟踪数据已保存到: {output_excel_path}")

        return True

    except Exception as e:
        import traceback
        print(f"处理视频 {input_video_path} 时发生严重错误: {e}")
        traceback.print_exc()
        return False


def batch_process_videos(
        txt_file_path,
        output_video_dir,
        output_gallery_dir,
        output_excel_dir,
        save_video=True,
        gpu_id=0  # 指定使用的GPU ID
):
    """
    从txt文件读取视频列表并批量处理。
    """
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_gallery_dir, exist_ok=True)
    os.makedirs(output_excel_dir, exist_ok=True)

    try:
        with open(txt_file_path, 'r') as f:
            video_files = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"错误: 找不到视频列表文件 {txt_file_path}")
        return

    successful_videos = 0
    total_videos = len(video_files)

    print(f"找到 {total_videos} 个视频进行处理。")

    for i, video_file in enumerate(video_files):
        print(f"\n--- 开始处理视频 {i + 1}/{total_videos}: {video_file} ---")
        if not os.path.exists(video_file):
            print(f"警告: 视频文件不存在，跳过: {video_file}")
            continue

        result = process_video(
            video_file,
            output_video_dir,
            output_gallery_dir,
            output_excel_dir,
            save_video,
            gpu_id
        )
        if result:
            successful_videos += 1
        print(f"--- 视频 {i + 1}/{total_videos} 处理完毕 ---")

    print(f"\n批量处理完成: 成功 {successful_videos}/{total_videos} 个视频。")


if __name__ == "__main__":
    # 在Windows上使用多处理时，建议使用 'spawn' 或 'forkserver'
    # 'spawn' 是最安全、最兼容的选择
    if os.name == 'posix':
        multiprocessing.set_start_method('fork', force=True)
    else:
        multiprocessing.set_start_method('spawn', force=True)

    # --- 用户配置区域 ---

    # 包含视频路径列表的txt文件
    INPUT_TXT_PATH = "input.txt"

    # 输出目录
    OUTPUT_VIDEO_DIR = r"output/dfm_track/video"
    OUTPUT_GALLERY_DIR = "output/dfm_track/gallery"
    OUTPUT_EXCEL_DIR = "output/dfm_track/excel"

    # 是否保存带跟踪框的视频
    SAVE_VIDEO = True

    # 指定使用的GPU ID
    GPU_ID = 0

    # --- 启动处理 ---
    batch_process_videos(
        INPUT_TXT_PATH,
        OUTPUT_VIDEO_DIR,
        OUTPUT_GALLERY_DIR,
        OUTPUT_EXCEL_DIR,
        save_video=SAVE_VIDEO,
        gpu_id=GPU_ID
    )
