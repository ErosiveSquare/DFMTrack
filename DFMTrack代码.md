```python
# DFM-Track: 动态特征流形跟踪器实现
# 修复了初始帧匹配问题
# ljy   8/20修复

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from typing import List, Tuple, Optional, Dict
import warnings

# 从 boxmot 导入基础跟踪模块
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator
from boxmot.trackers.dfmtrack.basetrack import BaseTrack, TrackState
from boxmot.appearance.reid_auto_backend import ReidAutoBackend

# 抑制警告以获得更清晰的输出
warnings.filterwarnings('ignore')


class DiagonalUKF:
    """
    对角无迹卡尔曼滤波器 (Diagonal Unscented Kalman Filter)
    专为高维特征跟踪优化的UKF实现。
    """

    def __init__(self, k_dim: int, q_base_std: float = 0.01, r_std: float = 0.1):
        """
        初始化对角UKF。
        """
        assert k_dim > 0, "状态空间维度必须为正"
        assert q_base_std > 0, "过程噪声标准差必须为正"
        assert r_std > 0, "测量噪声标准差必须为正"

        self.k_dim = k_dim

        # UKF缩放参数
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 3 - k_dim

        # 计算lambda参数
        self.lambda_ = self.alpha ** 2 * (self.k_dim + self.kappa) - self.k_dim

        # 计算sigma点的权重
        self.Wm = np.zeros(2 * self.k_dim + 1)
        self.Wc = np.zeros(2 * self.k_dim + 1)

        self.Wm[0] = self.lambda_ / (self.k_dim + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.k_dim + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.k_dim + 1):
            self.Wm[i] = 1 / (2 * (self.k_dim + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.k_dim + self.lambda_))

        # 噪声协方差
        self.Q_base = np.full(self.k_dim, q_base_std ** 2)
        self.Q = self.Q_base.copy()
        self.R = np.full(self.k_dim, r_std ** 2)

    def initiate(self, initial_feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用初始特征观测值初始化状态。"""
        assert len(initial_feature) == self.k_dim, \
            f"特征维度不匹配: 期望{self.k_dim}, 得到{len(initial_feature)}"

        mean = initial_feature.copy()
        covariance_diag = np.full(self.k_dim, 0.1)

        return mean, covariance_diag

    def generate_sigma_points(self, x: np.ndarray, P_diag: np.ndarray) -> np.ndarray:
        """为UKF生成sigma点。"""
        n_sigma = 2 * self.k_dim + 1
        sigmas = np.zeros((n_sigma, self.k_dim))

        sigmas[0] = x

        P_diag_safe = np.maximum(P_diag, 1e-10)
        sqrt_factor = np.sqrt((self.k_dim + self.lambda_) * P_diag_safe)

        for i in range(self.k_dim):
            sigmas[i + 1] = x.copy()
            sigmas[i + 1][i] += sqrt_factor[i]

            sigmas[self.k_dim + i + 1] = x.copy()
            sigmas[self.k_dim + i + 1][i] -= sqrt_factor[i]

        return sigmas

    def predict(self, x: np.ndarray, P_diag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """UKF的预测步骤。"""
        sigmas = self.generate_sigma_points(x, P_diag)
        sigmas_f = sigmas.copy()

        x_pred = np.dot(self.Wm, sigmas_f)

        y = sigmas_f - x_pred[np.newaxis, :]
        P_pred_diag = np.zeros(self.k_dim)
        for i in range(2 * self.k_dim + 1):
            P_pred_diag += self.Wc[i] * (y[i] ** 2)

        P_pred_diag += self.Q
        P_pred_diag = np.maximum(P_pred_diag, 1e-10)

        return x_pred, P_pred_diag

    def update(self, x_pred: np.ndarray, P_pred_diag: np.ndarray,
               z_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """UKF的更新步骤。"""
        innovation = z_obs - x_pred
        S_diag = P_pred_diag + self.R
        S_diag = np.maximum(S_diag, 1e-6)

        K_diag = P_pred_diag / S_diag
        K_diag = np.clip(K_diag, 0, 1)

        x_updated = x_pred + K_diag * innovation
        P_updated_diag = (1 - K_diag) * P_pred_diag
        P_updated_diag = np.maximum(P_updated_diag, 1e-6)

        return x_updated, P_updated_diag, innovation, S_diag

    def adapt_process_noise(self, innovation: np.ndarray, alpha_q: float = 0.1):
        """根据新息自适应调整过程噪声。"""
        self.Q = self.Q_base + alpha_q * (innovation ** 2)
        self.Q = np.clip(self.Q, self.Q_base * 0.5, self.Q_base * 10)


class DFMTracklet(BaseTrack):
    """DFM-Track系统中的单个跟踪单元。"""

    def __init__(self, det: np.ndarray, raw_feature: np.ndarray,
                 reduced_feature: np.ndarray, ukf: DiagonalUKF,
                 frame_id: int, track_id: Optional[int] = None):
        """初始化一个新的跟踪单元。"""
        super().__init__()

        # 物理属性
        self.tlbr = det[0:4].copy()
        self.conf = float(det[4])
        self.cls = int(det[5])
        self.det_ind = int(det[6]) if len(det) > 6 else -1

        # 特征属性 - 确保归一化
        raw_feature = raw_feature.copy()
        raw_feature_norm = np.linalg.norm(raw_feature)
        if raw_feature_norm > 0:
            raw_feature = raw_feature / raw_feature_norm

        self.raw_feature = raw_feature
        self.smooth_raw_feature = raw_feature.copy()
        self.alpha_ema = 0.9

        # UKF状态
        self.ukf = DiagonalUKF(ukf.k_dim, np.sqrt(ukf.Q_base[0]), np.sqrt(ukf.R[0]))
        self.mean, self.covariance_diag = self.ukf.initiate(reduced_feature)

        # 初始化时就设置预测状态为当前状态
        self.predicted_mean = self.mean.copy()
        self.predicted_cov_diag = self.covariance_diag.copy()

        # 轨迹管理
        if track_id is not None:
            self.track_id = track_id
        else:
            self.track_id = 0

        self.state = TrackState.New
        self.time_since_update = 0
        self.hits = 1

        # 帧跟踪
        self.frame_id = frame_id
        self.start_frame = frame_id

        # 兼容性
        self.score = self.conf
        self.is_activated = False

        # 历史观测
        self.history_observations = [self.tlbr.copy()]

    def predict(self):
        """使用UKF预测下一个状态。"""
        if self.mean is not None and self.covariance_diag is not None:
            self.predicted_mean, self.predicted_cov_diag = self.ukf.predict(
                self.mean, self.covariance_diag
            )
        else:
            # 如果没有有效状态，使用当前状态作为预测
            self.predicted_mean = self.mean
            self.predicted_cov_diag = self.covariance_diag

        self.time_since_update += 1

    def update(self, det: np.ndarray, raw_feature: np.ndarray,
               reduced_feature: np.ndarray, frame_id: int):
        """使用新的观测值更新跟踪单元。"""
        # 更新帧ID和统计信息
        self.frame_id = frame_id
        self.hits += 1
        self.time_since_update = 0

        # 更新物理属性
        self.tlbr = det[0:4].copy()
        self.conf = float(det[4])
        self.cls = int(det[5])
        self.score = self.conf

        # 添加到历史观测
        self.history_observations.append(self.tlbr.copy())
        if len(self.history_observations) > 30:
            self.history_observations.pop(0)

        # 确保特征归一化
        raw_feature = raw_feature.copy()
        raw_feature_norm = np.linalg.norm(raw_feature)
        if raw_feature_norm > 0:
            raw_feature = raw_feature / raw_feature_norm

        # UKF更新
        if self.predicted_mean is not None and self.predicted_cov_diag is not None:
            updated_mean, updated_cov_diag, innovation, S_diag = self.ukf.update(
                self.predicted_mean, self.predicted_cov_diag, reduced_feature
            )
            self.mean = updated_mean
            self.covariance_diag = updated_cov_diag
            self.ukf.adapt_process_noise(innovation, alpha_q=0.1)
        else:
            # 如果没有预测，重新初始化
            self.mean, self.covariance_diag = self.ukf.initiate(reduced_feature)

        # 更新预测状态为当前状态
        self.predicted_mean = self.mean.copy()
        self.predicted_cov_diag = self.covariance_diag.copy()

        # 更新平滑特征
        if len(self.smooth_raw_feature) != len(raw_feature):
            self.smooth_raw_feature = raw_feature.copy()
        else:
            self.smooth_raw_feature = (self.alpha_ema * self.smooth_raw_feature +
                                       (1 - self.alpha_ema) * raw_feature)
            smooth_norm = np.linalg.norm(self.smooth_raw_feature)
            if smooth_norm > 0:
                self.smooth_raw_feature = self.smooth_raw_feature / smooth_norm

        self.raw_feature = raw_feature

    # Add this method to the DFMTracklet class:

    def mark_removed(self):
        """标记轨迹为已移除状态。"""
        self.state = TrackState.Removed
        self.is_activated = False
        if hasattr(self, 'ukf'):
            # 清理UKF相关的大内存对象
            self.ukf = None
        # 可选：清理历史观测以节省内存
        if hasattr(self, 'history_observations') and len(self.history_observations) > 10:
            # 只保留最后10个观测用于可能的分析
            self.history_observations = self.history_observations[-10:]

    def activate(self, frame_id: int):
        """激活轨迹并分配ID。"""
        if not self.is_activated:
            self.track_id = self.next_id()
            self.is_activated = True
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, det: np.ndarray, raw_feature: np.ndarray,
                    reduced_feature: np.ndarray, frame_id: int):
        """重新激活一个丢失的轨迹。"""
        self.update(det, raw_feature, reduced_feature, frame_id)
        self.state = TrackState.Tracked
        self.is_activated = True

    @property
    def tlwh(self):
        """获取左上角-宽度-高度格式的边界框。"""
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def id(self):
        """获取轨迹ID。"""
        return self.track_id


class DFMTrack(BaseTracker):
    """动态特征流形跟踪器 (DFM-Track)"""

    def __init__(
            self,
            # 特征提取参数
            model_weights: Optional[str] = None,
            device: str = 'cpu',
            fp16: bool = False,
            per_class: bool = False,

            # 随机投影参数
            original_feature_dim: int = 1280,  # 修改为1280
            reduced_feature_dim: int = 64,

            # UKF参数
            ukf_q_std: float = 0.01,
            ukf_r_std: float = 0.1,

            # 数据关联参数
            mahalanobis_gate_thresh: float = 0.95,
            cosine_dist_thresh: float = 0.4,

            # 轨迹生命周期参数
            det_thresh: float = 0.3,
            max_age: int = 30,
            min_hits: int = 3,
            iou_threshold: float = 0.3,
            max_obs: int = 50
    ):
        """初始化DFM-Track。"""
        super().__init__(det_thresh, max_age, min_hits, iou_threshold, max_obs)

        # 参数验证
        assert original_feature_dim > 0, "原始特征维度必须为正"
        assert 0 < reduced_feature_dim <= original_feature_dim, \
            "降维维度必须为正且不大于原始维度"
        assert 0 < mahalanobis_gate_thresh < 1, "马氏距离门限必须在(0,1)之间"
        assert 0 < cosine_dist_thresh < 2, "余弦距离阈值必须在(0,2)之间"

        # 清除轨迹计数器
        BaseTrack.clear_count()

        # 特征维度
        self.N = original_feature_dim
        self.k = reduced_feature_dim

        # 设备和模型参数
        self.model_weights = model_weights
        self.device = device
        self.fp16 = fp16
        self.per_class = per_class

        self.with_reid = model_weights is not None
        if self.with_reid:
            try:
                self.reid_model = ReidAutoBackend(
                    weights=model_weights,
                    device=device,
                    half=fp16
                ).model
            except Exception as e:
                print(f"警告: 无法加载ReID模型权重 {model_weights}: {e}")
                print("将使用随机特征作为后备方案")
                self.with_reid = False
                self.reid_model = None
        else:
            self.reid_model = None

        print(f"DFM-Track: 已初始化, N={self.N}, k={self.k}, per_class={self.per_class}")

        # 初始化随机投影矩阵
        self.projection_matrix = None
        self.projection_scale = 1.0 / np.sqrt(self.k)
        self.actual_feature_dim = None

        # UKF模板
        self.ukf_template = DiagonalUKF(
            k_dim=self.k,
            q_base_std=ukf_q_std,
            r_std=ukf_r_std
        )

        # 数据关联阈值
        self.mahalanobis_thresh_sq = chi2.ppf(mahalanobis_gate_thresh, df=self.k)
        self.cosine_dist_thresh = cosine_dist_thresh

        # 轨迹列表
        self.tentative_tracks: List[DFMTracklet] = []
        self.confirmed_tracks: List[DFMTracklet] = []
        self.lost_tracks: List[DFMTracklet] = []
        self.removed_tracks: List[DFMTracklet] = []

    def _initialize_random_projection(self, feature_dim: int) -> np.ndarray:
        """使用Achlioptas方法初始化稀疏随机投影矩阵。"""
        try:
            print(f"[INFO] Initializing sparse random projection matrix ({self.k} x {feature_dim})...")

            # 验证输入
            if feature_dim <= 0:
                raise ValueError(f"Invalid feature dimension: {feature_dim}")
            if self.k <= 0:
                raise ValueError(f"Invalid target dimension: {self.k}")

            # 设置随机种子以获得可重复的结果（可选）
            # np.random.seed(42)

            # Achlioptas稀疏投影参数
            s = 3.0  # 稀疏度参数
            prob_zero = 1 - 1 / s
            prob_positive = 1 / (2 * s)
            prob_negative = 1 / (2 * s)

            # 生成随机矩阵
            rand_uniform = np.random.rand(self.k, feature_dim)
            projection = np.zeros((self.k, feature_dim), dtype=np.float32)

            # 根据概率分配值
            mask_positive = rand_uniform < prob_positive
            mask_negative = (rand_uniform >= prob_positive) & \
                            (rand_uniform < prob_positive + prob_negative)

            projection[mask_positive] = np.sqrt(s)
            projection[mask_negative] = -np.sqrt(s)

            # 统计信息
            n_positive = np.sum(mask_positive)
            n_negative = np.sum(mask_negative)
            n_zero = self.k * feature_dim - n_positive - n_negative
            sparsity = n_zero / (self.k * feature_dim)

            print(f"[INFO] Projection matrix initialized - "
                  f"Sparsity: {sparsity:.2%}, "
                  f"Positive: {n_positive}, "
                  f"Negative: {n_negative}, "
                  f"Zero: {n_zero}")

            # 验证结果
            if np.any(np.isnan(projection)) or np.any(np.isinf(projection)):
                raise ValueError("Invalid values in projection matrix")

            return projection

        except Exception as e:
            print(f"[ERROR] Failed to initialize projection matrix: {e}")
            # 使用标准高斯矩阵作为后备
            print(f"[INFO] Using standard Gaussian projection as fallback")
            return np.random.randn(self.k, feature_dim).astype(np.float32) / np.sqrt(self.k)



    def _project_features(self, features_raw: np.ndarray) -> np.ndarray:
        """将N维特征投影到k维空间。"""
        try:
            # 检查空输入
            if features_raw is None or features_raw.size == 0:
                return np.array([])

            # 确保输入是2D数组
            if features_raw.ndim == 1:
                features_raw = features_raw.reshape(1, -1)
                single_sample = True
            else:
                single_sample = False

            # 获取实际特征维度
            actual_dim = features_raw.shape[-1]

            # 如果投影矩阵未初始化或维度不匹配，重新初始化
            if self.projection_matrix is None or self.projection_matrix.shape[1] != actual_dim:
                if self.actual_feature_dim is not None and actual_dim != self.actual_feature_dim:
                    print(f"[WARNING] Feature dimension changed from {self.actual_feature_dim} to {actual_dim}")

                self.actual_feature_dim = actual_dim

                # 调整目标维度k，如果需要的话
                if self.k > actual_dim:
                    print(f"[WARNING] Reducing target dimension from {self.k} to {actual_dim}")
                    self.k = actual_dim
                    self.ukf_template = DiagonalUKF(
                        k_dim=self.k,
                        q_base_std=np.sqrt(self.ukf_template.Q_base[0]),
                        r_std=np.sqrt(self.ukf_template.R[0])
                    )

                # 初始化投影矩阵
                self.projection_matrix = self._initialize_random_projection(self.actual_feature_dim)
                self.projection_scale = 1.0 / np.sqrt(self.k)

                if self.frame_count == 1:
                    print(f"[INFO] Initialized projection matrix: {self.projection_matrix.shape}")

            # 执行投影
            projected = self.projection_scale * np.dot(self.projection_matrix, features_raw.T).T

            # 检查结果有效性
            if np.any(np.isnan(projected)) or np.any(np.isinf(projected)):
                print(f"[WARNING] Invalid values in projected features, using zeros")
                if single_sample:
                    return np.zeros(self.k)
                else:
                    return np.zeros((features_raw.shape[0], self.k))

            # 如果是单个样本，压缩维度
            if single_sample:
                projected = projected.squeeze()

            return projected

        except Exception as e:
            print(f"[ERROR] Feature projection failed: {e}")
            import traceback
            traceback.print_exc()

            # 返回零向量作为后备
            if features_raw.ndim == 1:
                return np.zeros(self.k)
            else:
                return np.zeros((features_raw.shape[0], self.k))



    def _compute_mahalanobis_distance_sq(self, track: DFMTracklet,
                                         det_feature: np.ndarray) -> float:
        """计算轨迹预测与检测之间的马氏距离平方。"""
        try:
            # 检查输入
            if det_feature is None or len(det_feature) == 0:
                return np.inf

            # 获取预测状态
            if track.predicted_mean is None or track.predicted_cov_diag is None:
                # 如果没有预测，使用当前状态
                if track.mean is None or track.covariance_diag is None:
                    if self.frame_count <= 5:
                        print(f"[WARNING] Track {track.track_id} has no valid state")
                    return np.inf
                predicted_mean = track.mean
                predicted_cov_diag = track.covariance_diag
            else:
                predicted_mean = track.predicted_mean
                predicted_cov_diag = track.predicted_cov_diag

            # 检查维度匹配
            if len(predicted_mean) != len(det_feature):
                if self.frame_count <= 5:
                    print(f"[WARNING] Dimension mismatch in Mahalanobis: "
                          f"predicted={len(predicted_mean)}, det={len(det_feature)}")
                return np.inf

            # 计算差异
            diff = det_feature - predicted_mean

            # 计算协方差（加上测量噪声）
            S_diag = predicted_cov_diag + self.ukf_template.R

            # 确保协方差为正
            S_diag = np.maximum(S_diag, 1e-10)

            # 检查是否有无效值
            if np.any(np.isnan(S_diag)) or np.any(np.isinf(S_diag)):
                if self.frame_count <= 5:
                    print(f"[WARNING] Invalid covariance values detected")
                return np.inf

            # 计算马氏距离平方
            dist_sq = np.sum(diff ** 2 / S_diag)

            # 检查结果有效性
            if np.isnan(dist_sq) or np.isinf(dist_sq):
                if self.frame_count <= 5:
                    print(f"[WARNING] Invalid Mahalanobis distance: {dist_sq}")
                return np.inf

            return dist_sq

        except Exception as e:
            if self.frame_count <= 5:
                print(f"[ERROR] Mahalanobis computation failed: {e}")
            return np.inf


    def _hierarchical_association(self, tracks: List[DFMTracklet],
                                  detections: List[Dict]) -> Tuple[List, List, List]:
        """使用马氏距离门控和余弦距离的分层数据关联。"""
        # 处理空列表情况
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        n_tracks = len(tracks)
        n_dets = len(detections)

        # 添加调试信息
        if self.frame_count % 100 == 0:
            print(f"[Association] Tracks: {n_tracks}, Detections: {n_dets}")

        # 初始化成本矩阵
        cost_matrix = np.full((n_tracks, n_dets), np.inf)

        # 第一层：马氏距离门控
        valid_pairs = 0
        mahalanobis_failures = 0
        cosine_failures = 0

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                try:
                    # 计算马氏距离
                    dist_sq = self._compute_mahalanobis_distance_sq(
                        track, det['reduced_feature']
                    )

                    # 应用门控
                    if dist_sq < self.mahalanobis_thresh_sq:
                        # 第二层：余弦距离
                        det_feature = det['raw_feature'].copy()
                        det_norm = np.linalg.norm(det_feature)
                        if det_norm > 0:
                            det_feature = det_feature / det_norm

                        # 检查特征维度匹配
                        if len(track.smooth_raw_feature) != len(det_feature):
                            if self.frame_count <= 5:  # 只在开始时打印
                                print(f"[WARNING] Feature dimension mismatch: "
                                      f"track={len(track.smooth_raw_feature)}, det={len(det_feature)}")
                            continue

                        # 计算余弦相似度
                        cosine_sim = np.clip(np.dot(track.smooth_raw_feature, det_feature), -1.0, 1.0)
                        cosine_dist = 1.0 - cosine_sim

                        # 应用第二层阈值
                        if cosine_dist < self.cosine_dist_thresh:
                            cost_matrix[i, j] = cosine_dist
                            valid_pairs += 1
                        else:
                            cosine_failures += 1
                    else:
                        mahalanobis_failures += 1

                except Exception as e:
                    if self.frame_count <= 5:
                        print(f"[ERROR] Association failed for track {i}, det {j}: {e}")
                    continue

        # 调试信息
        if self.frame_count % 100 == 0 and (mahalanobis_failures > 0 or cosine_failures > 0):
            print(f"[Association] Valid pairs: {valid_pairs}, "
                  f"Mahalanobis failures: {mahalanobis_failures}, "
                  f"Cosine failures: {cosine_failures}")

        # 如果没有有效的匹配对，直接返回
        if valid_pairs == 0:
            if self.frame_count % 50 == 0 and (n_tracks > 0 and n_dets > 0):
                print(f"[WARNING] No valid pairs found for {n_tracks} tracks and {n_dets} detections")
            return [], list(range(n_tracks)), list(range(n_dets))

        # 检查成本矩阵是否全为无穷大
        if np.all(np.isinf(cost_matrix)):
            if self.frame_count % 50 == 0:
                print(f"[WARNING] All costs are infinite")
            return [], list(range(n_tracks)), list(range(n_dets))

        # 使用匈牙利算法求解分配问题
        try:
            # 为了避免数值问题，将无穷大替换为一个大值
            cost_matrix_safe = cost_matrix.copy()
            max_finite = np.max(cost_matrix_safe[np.isfinite(cost_matrix_safe)], initial=1.0)
            cost_matrix_safe[np.isinf(cost_matrix_safe)] = max_finite * 1000

            row_indices, col_indices = linear_sum_assignment(cost_matrix_safe)

        except ValueError as e:
            print(f"[ERROR] Hungarian algorithm failed: {e}")
            print(f"Cost matrix shape: {cost_matrix.shape}")
            print(f"Valid pairs: {valid_pairs}")
            print(f"Cost matrix finite values: {np.sum(np.isfinite(cost_matrix))}")
            return [], list(range(n_tracks)), list(range(n_dets))
        except Exception as e:
            print(f"[ERROR] Unexpected error in Hungarian algorithm: {e}")
            return [], list(range(n_tracks)), list(range(n_dets))

        # 筛选有效匹配（只保留原始成本矩阵中有限的匹配）
        matched_indices = []
        matched_track_set = set()
        matched_det_set = set()

        for row, col in zip(row_indices, col_indices):
            if np.isfinite(cost_matrix[row, col]) and cost_matrix[row, col] < self.cosine_dist_thresh:
                matched_indices.append((row, col))
                matched_track_set.add(row)
                matched_det_set.add(col)

        # 找到未匹配项
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_det_set]

        # 调试信息
        if self.frame_count % 100 == 0 and len(matched_indices) > 0:
            print(f"[Association] Matched: {len(matched_indices)}, "
                  f"Unmatched tracks: {len(unmatched_tracks)}, "
                  f"Unmatched dets: {len(unmatched_dets)}")

        return matched_indices, unmatched_tracks, unmatched_dets


    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray,
               embs: Optional[np.ndarray] = None) -> np.ndarray:
        """使用新的检测结果更新跟踪器。"""
        # 输入验证
        self.check_inputs(dets, img)
        self.frame_count += 1

        # 添加调试信息
        if self.frame_count % 50 == 0:
            print(f"[Frame {self.frame_count}] Active tracks - Confirmed: {len(self.confirmed_tracks)}, "
                  f"Tentative: {len(self.tentative_tracks)}")

        # 添加检测索引
        if len(dets) > 0:
            det_indices = np.arange(len(dets)).reshape(-1, 1)
            dets = np.hstack([dets, det_indices])

        # 提取或生成特征
        if self.with_reid:
            if embs is not None:
                raw_features = embs.copy()
            else:
                if len(dets) > 0:
                    bboxes = dets[:, 0:4]
                    try:
                        raw_features = self.reid_model.get_features(bboxes, img)
                        if isinstance(raw_features, torch.Tensor):
                            raw_features = raw_features.cpu().numpy()
                        if self.frame_count == 1:
                            print(f"ReID模型输出特征维度: {raw_features.shape}")
                    except Exception as e:
                        print(f"ReID特征提取失败: {e}")
                        raw_features = np.random.randn(len(dets), self.N)
                else:
                    raw_features = np.array([])

            # 确保特征归一化
            if len(raw_features) > 0:
                if raw_features.ndim == 1:
                    raw_features = raw_features.reshape(1, -1)
                for i in range(len(raw_features)):
                    norm = np.linalg.norm(raw_features[i])
                    if norm > 0:
                        raw_features[i] = raw_features[i] / norm
        else:
            if len(dets) > 0:
                feature_dim = self.actual_feature_dim if self.actual_feature_dim else 1280
                raw_features = np.random.randn(len(dets), feature_dim)
                for i in range(len(raw_features)):
                    norm = np.linalg.norm(raw_features[i])
                    if norm > 0:
                        raw_features[i] = raw_features[i] / norm
            else:
                raw_features = np.array([])

        # 投影特征到降维空间
        if len(raw_features) > 0:
            reduced_features = self._project_features(raw_features)
            if self.frame_count == 1:
                print(f"投影后特征维度: {reduced_features.shape}")
        else:
            reduced_features = np.array([])

        # 封装检测结果
        current_detections = []
        for i in range(len(dets)):
            if i < len(raw_features) and i < len(reduced_features):
                current_detections.append({
                    'det': dets[i],
                    'raw_feature': raw_features[i],
                    'reduced_feature': reduced_features[i]
                })

        # 对所有活跃轨迹进行预测
        for track in self.confirmed_tracks:
            track.predict()
        for track in self.tentative_tracks:
            track.predict()

        # 分层数据关联
        # 1. 匹配确认轨迹
        matched_confirmed, unmatched_confirmed, unmatched_dets_after_confirmed = \
            self._hierarchical_association(self.confirmed_tracks, current_detections)

        # 2. 用剩余检测匹配试探轨迹
        remaining_dets = [current_detections[i] for i in unmatched_dets_after_confirmed]
        matched_tentative, unmatched_tentative, final_unmatched_dets_indices = \
            self._hierarchical_association(self.tentative_tracks, remaining_dets)

        # 添加调试信息
        if len(matched_tentative) > 0:
            print(f"[Frame {self.frame_count}] Matched tentative: {len(matched_tentative)}, "
                  f"Total tentative tracks: {len(self.tentative_tracks)}")
            for track_idx, det_idx in matched_tentative:
                if track_idx >= len(self.tentative_tracks):
                    print(
                        f"ERROR: Invalid track_idx {track_idx}, tentative_tracks length: {len(self.tentative_tracks)}")

        # 更新确认轨迹
        for track_idx, det_idx in matched_confirmed:
            if track_idx < len(self.confirmed_tracks):
                track = self.confirmed_tracks[track_idx]
                det_info = current_detections[det_idx]
                track.update(
                    det_info['det'],
                    det_info['raw_feature'],
                    det_info['reduced_feature'],
                    self.frame_count
                )
            else:
                print(f"WARNING: Invalid confirmed track index {track_idx}")

        # 更新试探轨迹并收集要确认的轨迹
        tentative_to_confirm = []
        for track_idx, det_idx in matched_tentative:
            if track_idx < len(self.tentative_tracks) and det_idx < len(remaining_dets):
                track = self.tentative_tracks[track_idx]
                det_info = remaining_dets[det_idx]
                track.update(
                    det_info['det'],
                    det_info['raw_feature'],
                    det_info['reduced_feature'],
                    self.frame_count
                )
                if track.hits >= self.min_hits:
                    track.activate(self.frame_count)
                    tentative_to_confirm.append(track)
            else:
                print(f"WARNING: Invalid tentative indices - track_idx: {track_idx}/{len(self.tentative_tracks)}, "
                      f"det_idx: {det_idx}/{len(remaining_dets)}")

        # 使用对象引用而不是索引来处理未匹配的确认轨迹
        tracks_to_remove_from_confirmed = []
        for track_idx in unmatched_confirmed:
            if track_idx < len(self.confirmed_tracks):
                track = self.confirmed_tracks[track_idx]
                if track.time_since_update > self.max_age:
                    track.mark_removed()
                    self.lost_tracks.append(track)
                    tracks_to_remove_from_confirmed.append(track)

        # 使用对象引用来处理未匹配的试探轨迹
        unmatched_tentative_tracks = []
        for track_idx in unmatched_tentative:
            if track_idx < len(self.tentative_tracks):
                unmatched_tentative_tracks.append(self.tentative_tracks[track_idx])

        # 现在安全地修改列表
        # 1. 将确认的轨迹从tentative移到confirmed
        for track in tentative_to_confirm:
            if track in self.tentative_tracks:
                self.tentative_tracks.remove(track)
                self.confirmed_tracks.append(track)

        # 2. 移除失效的试探轨迹
        tentative_to_remove = []
        for track in unmatched_tentative_tracks:
            if track.time_since_update > self.min_hits:
                track.mark_removed()
                tentative_to_remove.append(track)

        for track in tentative_to_remove:
            if track in self.tentative_tracks:
                self.tentative_tracks.remove(track)

        # 3. 移除失效的确认轨迹
        for track in tracks_to_remove_from_confirmed:
            if track in self.confirmed_tracks:
                self.confirmed_tracks.remove(track)

        # 为未匹配的高置信度检测创建新轨迹
        final_unmatched_dets = [remaining_dets[i] for i in final_unmatched_dets_indices
                                if i < len(remaining_dets)]
        for det_info in final_unmatched_dets:
            if det_info['det'][4] >= self.det_thresh:
                new_track = DFMTracklet(
                    det_info['det'],
                    det_info['raw_feature'],
                    det_info['reduced_feature'],
                    self.ukf_template,
                    self.frame_count
                )
                self.tentative_tracks.append(new_track)

        # 清理已标记为移除的轨迹（额外的安全检查）
        self.confirmed_tracks = [t for t in self.confirmed_tracks
                                 if t.state != TrackState.Removed]

        # 定期清理内存
        if self.frame_count % 100 == 0:
            if len(self.lost_tracks) > 100:
                self.lost_tracks = self.lost_tracks[-50:]
            if len(self.removed_tracks) > 1000:
                self.removed_tracks = self.removed_tracks[-500:]
            print(f"[Frame {self.frame_count}] Memory cleanup - Lost tracks: {len(self.lost_tracks)}, "
                  f"Removed tracks: {len(self.removed_tracks)}")

        # 生成输出
        outputs = []
        for track in self.confirmed_tracks:
            if track.state == TrackState.Tracked and track.is_activated:
                output = np.concatenate([
                    track.tlbr,
                    [track.track_id, track.conf, track.cls, track.det_ind]
                ])
                outputs.append(output)

        # 更新活跃轨迹列表
        self.active_tracks = self.confirmed_tracks.copy()

        if len(outputs) > 0:
            return np.array(outputs)
        else:
            return np.empty((0, 8))



    def reset(self):
        """重置跟踪器状态。"""
        BaseTrack.clear_count()
        self.frame_count = 0
        self.tentative_tracks.clear()
        self.confirmed_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.active_tracks.clear()
        self.projection_matrix = None
        self.actual_feature_dim = None
```

