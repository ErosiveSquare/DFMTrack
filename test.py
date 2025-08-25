"""
测试ReID模型和特征提取的脚本
用于诊断clip_zhz.pt权重文件的问题
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from boxmot.appearance.reid_auto_backend import ReidAutoBackend

def test_reid_model():
    """测试ReID模型是否正常工作"""

    print("=" * 60)
    print("ReID模型测试")
    print("=" * 60)

    # 1. 检查权重文件
    weights_path = Path('weights/clip_zhz.pt')
    if not weights_path.exists():
        print(f"❌ 权重文件不存在: {weights_path}")
        return

    print(f"✓ 权重文件存在: {weights_path}")
    print(f"  文件大小: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 2. 加载模型
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"\n使用设备: {device}")

        # 修复：传递Path对象而不是字符串
        reid_backend = ReidAutoBackend(
            weights=weights_path,  # 使用Path对象
            device=device,
            half=False
        )
        reid_model = reid_backend.model
        print("✓ ReID模型加载成功")

    except Exception as e:
        print(f"❌ ReID模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 创建测试图像和边界框
    print("\n测试特征提取...")

    # 创建一个简单的测试图像 (720p)
    test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # 创建几个测试边界框
    test_bboxes = np.array([
        [100, 100, 200, 300],  # person 1
        [300, 150, 400, 350],  # person 2
        [500, 200, 600, 400],  # person 3
    ], dtype=np.float32)

    print(f"测试图像尺寸: {test_img.shape}")
    print(f"测试边界框数量: {len(test_bboxes)}")

    # 4. 提取特征
    try:
        features = reid_model.get_features(test_bboxes, test_img)

        # 转换为numpy
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        print(f"\n✓ 特征提取成功!")
        print(f"  特征形状: {features.shape}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  特征数据类型: {features.dtype}")

        # 检查特征值范围
        print(f"\n特征统计:")
        print(f"  最小值: {features.min():.4f}")
        print(f"  最大值: {features.max():.4f}")
        print(f"  均值: {features.mean():.4f}")
        print(f"  标准差: {features.std():.4f}")

        # 测试特征归一化
        norms = np.linalg.norm(features, axis=1)
        print(f"\n特征范数:")
        print(f"  范数: {norms}")

        # 归一化特征
        normalized_features = features / norms[:, np.newaxis]
        normalized_norms = np.linalg.norm(normalized_features, axis=1)
        print(f"  归一化后范数: {normalized_norms}")

        # 5. 测试特征相似度
        print(f"\n特征相似度测试:")
        # 计算特征之间的余弦相似度
        for i in range(len(normalized_features)):
            for j in range(i+1, len(normalized_features)):
                similarity = np.dot(normalized_features[i], normalized_features[j])
                print(f"  特征{i} vs 特征{j}: {similarity:.4f}")

        print("\n✅ ReID模型测试完成，一切正常!")
        print(f"\n重要信息：")
        print(f"  - ReID模型输出维度: {features.shape[1]}维")
        print(f"  - 建议在DFMTrack中设置 original_feature_dim={features.shape[1]}")

        return features.shape[1]

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_video():
    """使用真实视频测试"""
    video_path = "video/test.mp4"

    if not Path(video_path).exists():
        print(f"\n⚠️ 测试视频不存在: {video_path}")
        return

    print("\n" + "=" * 60)
    print("使用真实视频测试")
    print("=" * 60)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取视频帧")
        cap.release()
        return

    print(f"视频帧尺寸: {frame.shape}")

    # 加载YOLO模型进行检测
    try:
        from ultralytics import YOLO
        model = YOLO('weights/yolov8m.pt')

        # 检测
        results = model(frame, conf=0.3, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        # 筛选person类别(假设类别ID为0)
        person_dets = detections[detections[:, 5] == 0]

        if len(person_dets) > 0:
            print(f"检测到 {len(person_dets)} 个人")

            # 使用ReID提取特征
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            # 修复：使用Path对象
            reid_backend = ReidAutoBackend(
                weights=Path('weights/clip_zhz.pt'),  # 使用Path对象
                device=device,
                half=False
            )
            reid_model = reid_backend.model

            bboxes = person_dets[:, :4]
            features = reid_model.get_features(bboxes, frame)

            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

            print(f"提取的特征维度: {features.shape}")

        else:
            print("未检测到人")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    cap.release()


if __name__ == "__main__":
    # 运行测试
    feature_dim = test_reid_model()

    # 如果需要，使用真实视频测试
    test_with_real_video()

    # 给出建议
    if feature_dim:
        print("\n" + "=" * 60)
        print("建议的DFMTrack配置:")
        print("=" * 60)
        print(f"""
tracker = DFMTrack(
    model_weights=Path('weights/clip_zhz.pt'),
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    fp16=False,
    original_feature_dim={feature_dim},  # 使用实际的特征维度
    reduced_feature_dim=64,
    ukf_q_std=0.01,
    ukf_r_std=0.1,
    mahalanobis_gate_thresh=0.95,
    cosine_dist_thresh=0.4,
    det_thresh=0.3,
    max_age=30,
    min_hits=3
)
        """)