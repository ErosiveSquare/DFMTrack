# DFM-Track: 动态特征流形跟踪器实现
# 一个在多维特征空间中运行的纯外观跟踪器
# 作者: LJY 0726-0815

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from typing import List, Tuple, Optional, Dict
import warnings

# 从 boxmot 导入基础跟踪模块
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState

# 抑制警告以获得更清晰的输出
warnings.filterwarnings('ignore')


class DiagonalUKF:
    """
    对角无迹卡尔曼滤波器 (Diagonal Unscented Kalman Filter)，为高维特征跟踪进行了优化。

    此实现利用对角协方差假设将计算复杂度从 O(k²) 降低到 O(k)，
    使其适用于高维特征空间中的实时跟踪。
    """

    def __init__(self, k_dim: int, q_base_std: float = 0.01, r_std: float = 0.1):
        """
        初始化对角 UKF。

        Args:
            k_dim: 状态空间的维度 (降维后的特征维度)
            q_base_std: 过程噪声的基础标准差
            r_std: 测量噪声的标准差
        """
        self.k_dim = k_dim

        # UKF 缩放参数
        self.alpha = 1e-3  # 控制 sigma 点在均值周围的分布范围
        self.beta = 2.0  # 用于合并关于分布的先验知识
        self.kappa = 3 - k_dim  # 次要缩放参数

        # 计算 lambda
        self.lambda_ = self.alpha ** 2 * (self.k_dim + self.kappa) - self.k_dim

        # 计算 sigma 点的权重
        self.Wm = np.zeros(2 * self.k_dim + 1)  # 用于计算均值的权重
        self.Wc = np.zeros(2 * self.k_dim + 1)  # 用于计算协方差的权重

        self.Wm[0] = self.lambda_ / (self.k_dim + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.k_dim + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.k_dim + 1):
            self.Wm[i] = 1 / (2 * (self.k_dim + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.k_dim + self.lambda_))

        # 过程噪声和测量噪声 (对角阵，以向量形式存储)
        self.Q_base = np.full(self.k_dim, q_base_std ** 2)
        self.Q = self.Q_base.copy()  # 自适应过程噪声
        self.R = np.full(self.k_dim, r_std ** 2)  # 测量噪声

    def initiate(self, initial_feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用初始特征观测值初始化状态。

        Args:
            initial_feature: k 维的初始特征向量

        Returns:
            mean: 初始状态均值
            covariance_diag: 初始对角协方差
        """
        mean = initial_feature.copy()
        # 以中等不确定性开始
        covariance_diag = np.full(self.k_dim, 0.1)
        return mean, covariance_diag

    def generate_sigma_points(self, x: np.ndarray, P_diag: np.ndarray) -> np.ndarray:
        """
        为 UKF 生成 sigma 点。

        Args:
            x: 状态均值
            P_diag: 协方差矩阵的对角线元素

        Returns:
            Sigma 点矩阵 (2k+1 x k)
        """
        n_sigma = 2 * self.k_dim + 1
        sigmas = np.zeros((n_sigma, self.k_dim))

        # 第一个 sigma 点是均值
        sigmas[0] = x

        # 利用对角假设计算 (k_dim + lambda) * P 的平方根
        sqrt_factor = np.sqrt((self.k_dim + self.lambda_) * P_diag)

        # 生成其余的 sigma 点
        for i in range(self.k_dim):
            sigmas[i + 1] = x.copy()
            sigmas[i + 1][i] += sqrt_factor[i]

            sigmas[self.k_dim + i + 1] = x.copy()
            sigmas[self.k_dim + i + 1][i] -= sqrt_factor[i]

        return sigmas

    def predict(self, x: np.ndarray, P_diag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF 的预测步骤，使用近似恒等模型。

        Args:
            x: 当前状态均值
            P_diag: 当前对角协方差

        Returns:
            x_pred: 预测的状态均值
            P_pred_diag: 预测的对角协方差
        """
        # 生成 sigma 点
        sigmas = self.generate_sigma_points(x, P_diag)

        # 将 sigma 点通过运动模型（在我们的例子中是恒等模型）
        # 对于近似恒等模型，f(x) = x
        sigmas_f = sigmas.copy()

        # 计算预测均值
        x_pred = np.dot(self.Wm, sigmas_f)

        # 计算预测协方差
        y = sigmas_f - x_pred[np.newaxis, :]
        P_pred_diag = np.zeros(self.k_dim)
        for i in range(2 * self.k_dim + 1):
            P_pred_diag += self.Wc[i] * (y[i] ** 2)

        # 添加过程噪声
        P_pred_diag += self.Q

        return x_pred, P_pred_diag

    def update(self, x_pred: np.ndarray, P_pred_diag: np.ndarray,
               z_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        UKF 的更新步骤。

        Args:
            x_pred: 预测的状态均值
            P_pred_diag: 预测的对角协方差
            z_obs: 观测到的测量值

        Returns:
            x_updated: 更新后的状态均值
            P_updated_diag: 更新后的对角协方差
            innovation: 新息向量 (y)
            S_diag: 新息协方差的对角线
        """
        # 对于观测模型 h(x) = x (恒等模型)
        z_pred = x_pred

        # 新息协方差: S = P_pred + R
        S_diag = P_pred_diag + self.R

        # 卡尔曼增益: K = P_pred * S^(-1)，针对对角矩阵
        K_diag = P_pred_diag / (S_diag + 1e-10)  # 添加一个小的 epsilon 以保证数值稳定性

        # 新息
        innovation = z_obs - z_pred

        # 更新状态
        x_updated = x_pred + K_diag * innovation

        # 更新协方差: P = (I - K*H) * P_pred = (1 - K) * P_pred，针对对角矩阵
        P_updated_diag = (1 - K_diag) * P_pred_diag

        return x_updated, P_updated_diag, innovation, S_diag

    def adapt_process_noise(self, innovation: np.ndarray, alpha_q: float = 0.1):
        """
        根据新息的幅度自适应调整过程噪声。

        Args:
            innovation: 更新步骤中得到的新息向量
            alpha_q: 自适应速率
        """
        # 当新息较大时，增加过程噪声
        self.Q = self.Q_base + alpha_q * (innovation ** 2)
        # 将其限制在合理范围内
        self.Q = np.clip(self.Q, self.Q_base * 0.5, self.Q_base * 10)


class DFMTracklet(BaseTrack):
    """
    DFM-Track 系统中的单个跟踪单元（Tracklet）。
    继承自 BaseTrack 以保持与 boxmot 框架的兼容性。
    """

    def __init__(self, det: np.ndarray, raw_feature: np.ndarray,
                 reduced_feature: np.ndarray, ukf: DiagonalUKF,
                 frame_id: int, track_id: Optional[int] = None):
        """
        初始化一个新的跟踪单元。

        Args:
            det: 检测框数组 [x1, y1, x2, y2, conf, cls, det_idx]
            raw_feature: 原始 N 维特征向量
            reduced_feature: 降维后的 k 维特征向量
            ukf: 用于此轨迹的 UKF 实例
            frame_id: 当前帧 ID
            track_id: 可选的轨迹 ID (用于重识别)
        """
        super().__init__()

        # 物理属性 (用于输出)
        self.tlbr = det[0:4]  # 左上角-右下角坐标格式
        self.conf = det[4]
        self.cls = int(det[5])
        self.det_ind = int(det[6]) if len(det) > 6 else -1

        # 特征属性
        self.raw_feature = raw_feature.copy()
        self.smooth_raw_feature = raw_feature.copy()  # EMA (指数移动平均) 平滑后的特征
        self.alpha_ema = 0.9  # EMA 平滑因子

        # 确保特征被归一化
        self.raw_feature /= (np.linalg.norm(self.raw_feature) + 1e-10)
        self.smooth_raw_feature /= (np.linalg.norm(self.smooth_raw_feature) + 1e-10)

        # UKF 状态
        self.ukf = DiagonalUKF(ukf.k_dim, ukf.Q_base[0] ** 0.5, ukf.R[0] ** 0.5)
        self.mean, self.covariance_diag = self.ukf.initiate(reduced_feature)
        self.predicted_mean = None
        self.predicted_cov_diag = None

        # 轨迹管理
        if track_id is not None:
            self.track_id = track_id
        else:
            self.track_id = 0

        self.state = TrackState.New  # 初始状态为试探性 (New)
        self.time_since_update = 0
        self.hits = 1

        # 帧跟踪
        self.frame_id = frame_id
        self.start_frame = frame_id

        # 为了与 boxmot 兼容
        self.score = self.conf
        self.is_activated = False

    def predict(self):
        """使用 UKF 预测下一个状态。"""
        self.predicted_mean, self.predicted_cov_diag = self.ukf.predict(
            self.mean, self.covariance_diag
        )
        self.time_since_update += 1

    def update(self, det: np.ndarray, raw_feature: np.ndarray,
               reduced_feature: np.ndarray, frame_id: int):
        """
        使用新的观测值更新跟踪单元。

        Args:
            det: 新的检测框
            raw_feature: 新的原始特征
            reduced_feature: 新的降维后特征
            frame_id: 当前帧 ID
        """
        # 更新帧 ID
        self.frame_id = frame_id

        # 更新跟踪统计信息
        self.hits += 1
        self.time_since_update = 0

        # 更新物理属性
        self.tlbr = det[0:4]
        self.conf = det[4]
        self.cls = int(det[5])
        self.score = self.conf

        # UKF 更新
        if self.predicted_mean is not None:
            updated_mean, updated_cov_diag, innovation, S_diag = self.ukf.update(
                self.predicted_mean, self.predicted_cov_diag, reduced_feature
            )
            self.mean = updated_mean
            self.covariance_diag = updated_cov_diag

            # 自适应过程噪声
            self.ukf.adapt_process_noise(innovation)
        else:
            # 如果没有进行预测，则重新初始化
            self.mean, self.covariance_diag = self.ukf.initiate(reduced_feature)

        # 使用 EMA 平滑更新原始特征
        self.raw_feature = raw_feature.copy()
        self.raw_feature /= (np.linalg.norm(self.raw_feature) + 1e-10)

        self.smooth_raw_feature = (self.alpha_ema * self.smooth_raw_feature +
                                   (1 - self.alpha_ema) * self.raw_feature)
        self.smooth_raw_feature /= (np.linalg.norm(self.smooth_raw_feature) + 1e-10)

    def activate(self, frame_id: int):
        """激活轨迹并分配 ID。"""
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


class DFMTrack(BaseTracker):
    """
    动态特征流形跟踪器 (DFM-Track)

    一个纯外观的多目标跟踪器，完全在高维特征空间中运行，
    使用非线性时间滤波来保持跨帧的身份一致性。
    """

    def __init__(
            self,
            # 特征提取参数
            model_weights: Optional[str] = None,
            device: str = 'cpu',
            fp16: bool = False,

            # 随机投影参数
            original_feature_dim: int = 512,
            reduced_feature_dim: int = 64,

            # UKF 参数
            ukf_q_std: float = 0.01,
            ukf_r_std: float = 0.1,

            # 数据关联参数
            mahalanobis_gate_thresh: float = 0.95,
            cosine_dist_thresh: float = 0.4,

            # 轨迹生命周期参数
            det_thresh: float = 0.3,
            max_age: int = 30,
            min_hits: int = 3,
            iou_threshold: float = 0.3,  # 未使用，但为保持兼容性而保留
            max_obs: int = 50
    ):
        """
        初始化 DFM-Track。

        Args:
            model_weights: ReID 模型权重的路径
            device: 用于计算的设备
            fp16: 是否使用半精度
            original_feature_dim: 原始 ReID 特征的维度
            reduced_feature_dim: 随机投影后的维度
            ukf_q_std: UKF 的过程噪声标准差
            ukf_r_std: UKF 的测量噪声标准差
            mahalanobis_gate_thresh: 用于门控的卡方分布百分位数
            cosine_dist_thresh: 余弦距离匹配的阈值
            det_thresh: 检测置信度阈值
            max_age: 保持丢失轨迹的最大帧数
            min_hits: 确认轨迹所需的最小命中次数
            iou_threshold: IOU 阈值 (为保持兼容性而保留)
            max_obs: 存储的最大观测数
        """
        super().__init__(det_thresh, max_age, min_hits, iou_threshold, max_obs)

        # 清除轨迹计数器
        BaseTrack.clear_count()

        # 特征维度
        self.N = original_feature_dim
        self.k = reduced_feature_dim

        # 初始化 ReID 模型 (占位符 - 实际实现中会加载模型)
        self.model_weights = model_weights
        self.device = device
        self.fp16 = fp16
        print(f"DFM-Track: 已初始化, N={self.N}, k={self.k}")

        # 初始化随机投影矩阵 (一次性初始化)
        self.projection_matrix = self._initialize_random_projection()
        self.projection_scale = 1.0 / np.sqrt(self.k)

        # UKF 模板
        self.ukf_template = DiagonalUKF(
            k_dim=self.k,
            q_base_std=ukf_q_std,
            r_std=ukf_r_std
        )

        # 数据关联阈值
        self.mahalanobis_thresh_sq = chi2.ppf(mahalanobis_gate_thresh, df=self.k)
        self.cosine_dist_thresh = cosine_dist_thresh

        # 轨迹列表
        self.tentative_tracks: List[DFMTracklet] = []  # 试探性轨迹
        self.confirmed_tracks: List[DFMTracklet] = []  # 已确认的轨迹
        self.lost_tracks: List[DFMTracklet] = []  # 已丢失的轨迹
        self.removed_tracks: List[DFMTracklet] = []  # 已移除的轨迹

    def _initialize_random_projection(self) -> np.ndarray:
        """
        使用 Achlioptas 方法初始化稀疏随机投影矩阵。

        Returns:
            形状为 (k, N) 的随机投影矩阵
        """
        print(f"正在初始化稀疏随机投影矩阵 ({self.k} x {self.N})...")

        # Achlioptas 的稀疏随机投影方法
        s = 3.0
        prob_zero = 1 - 1 / s
        prob_positive = 1 / (2 * s)
        prob_negative = 1 / (2 * s)

        # 生成随机矩阵
        rand_uniform = np.random.rand(self.k, self.N)
        projection = np.zeros((self.k, self.N))

        # 根据概率填充矩阵
        mask_positive = rand_uniform < prob_positive
        mask_negative = (rand_uniform >= prob_positive) & (rand_uniform < prob_positive + prob_negative)

        projection[mask_positive] = np.sqrt(s)
        projection[mask_negative] = -np.sqrt(s)

        return projection

    def _project_features(self, features_raw: np.ndarray) -> np.ndarray:
        """
        将 N 维特征投影到 k 维空间。

        Args:
            features_raw: 形状为 (n_samples, N) 或 (N,) 的原始特征

        Returns:
            形状为 (n_samples, k) 或 (k,) 的投影后特征
        """
        if features_raw.ndim == 1:
            features_raw = features_raw.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # 应用投影: f_reduced = (1/√k) * R * f_raw^T
        projected = self.projection_scale * np.dot(self.projection_matrix, features_raw.T).T

        if single_sample:
            projected = projected.squeeze()

        return projected

    def _compute_mahalanobis_distance_sq(self, track: DFMTracklet,
                                         det_feature: np.ndarray) -> float:
        """
        计算轨迹预测与检测之间的马氏距离平方。

        Args:
            track: 带有预测状态的轨迹
            det_feature: 降维空间中的检测特征

        Returns:
            马氏距离的平方
        """
        if track.predicted_mean is None:
            return np.inf

        # 新息
        diff = det_feature - track.predicted_mean

        # 新息协方差: S = P_pred + R
        S_diag = track.predicted_cov_diag + self.ukf_template.R

        # 马氏距离平方: d^2 = y^T * S^(-1) * y
        # 对于对角矩阵: sum(y^2 / S_diag)
        dist_sq = np.sum(diff ** 2 / (S_diag + 1e-10))

        return dist_sq

    def _hierarchical_association(self, tracks: List[DFMTracklet],
                                  detections: List[Dict]) -> Tuple[List, List, List]:
        """
        使用马氏距离门控和余弦距离的分层数据关联。

        Args:
            tracks: 待匹配的轨迹列表
            detections: 检测结果字典的列表

        Returns:
            matched_indices: (track_idx, det_idx) 对的列表
            unmatched_tracks: 未匹配轨迹的索引列表
            unmatched_dets: 未匹配检测的索引列表
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        n_tracks = len(tracks)
        n_dets = len(detections)

        # 用无穷大初始化成本矩阵
        cost_matrix = np.full((n_tracks, n_dets), np.inf)

        # 第一层：马氏距离门控
        for i, track in enumerate(tracks):
            if track.predicted_mean is None:
                continue

            for j, det in enumerate(detections):
                # 计算马氏距离
                dist_sq = self._compute_mahalanobis_distance_sq(
                    track, det['reduced_feature']
                )

                # 应用门控
                if dist_sq < self.mahalanobis_thresh_sq:
                    # 第二层：余弦距离
                    # 使用平滑后的特征进行更稳定的匹配
                    cosine_sim = np.dot(track.smooth_raw_feature, det['raw_feature'])
                    cosine_dist = 1.0 - cosine_sim

                    # 应用第二层阈值
                    if cosine_dist < self.cosine_dist_thresh:
                        cost_matrix[i, j] = cosine_dist

        # 使用匈牙利算法解决分配问题
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 筛选有效匹配
        matched_indices = []
        matched_track_set = set()
        matched_det_set = set()

        for row, col in zip(row_indices, col_indices):
            if np.isfinite(cost_matrix[row, col]):
                matched_indices.append((row, col))
                matched_track_set.add(row)
                matched_det_set.add(col)

        # 找到未匹配项
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_det_set]

        return matched_indices, unmatched_tracks, unmatched_dets

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray,
               embs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用新的检测结果更新跟踪器。

        Args:
            dets: 形状为 (n, 6)，格式为 [x1, y1, x2, y2, conf, cls] 的检测数组
            img: 当前帧 (在纯外观跟踪中未使用)
            embs: 预先计算好的嵌入特征，形状为 (n, N)

        Returns:
            形状为 (m, 8)，格式为 [x1, y1, x2, y2, id, conf, cls, det_idx] 的跟踪结果
        """
        self.check_inputs(dets, img)
        self.frame_count += 1

        # 添加检测索引
        if len(dets) > 0:
            det_indices = np.arange(len(dets)).reshape(-1, 1)
            dets = np.hstack([dets, det_indices])

        # 提取或生成特征
        if embs is None:
            # 在实际实现中，会使用 ReID 模型提取特征
            # 此处为演示目的，生成随机特征
            raw_features = np.random.randn(len(dets), self.N) if len(dets) > 0 else np.array([])
            if len(raw_features) > 0:
                # 归一化特征
                raw_features = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-10)
        else:
            raw_features = embs
            # 确保归一化
            if len(raw_features) > 0:
                raw_features = raw_features / (np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-10)

        # 将特征投影到降维空间
        reduced_features = self._project_features(raw_features) if len(raw_features) > 0 else np.array([])

        # 封装检测结果
        current_detections = []
        for i in range(len(dets)):
            current_detections.append({
                'det': dets[i],
                'raw_feature': raw_features[i],
                'reduced_feature': reduced_features[i]
            })

        # 对所有轨迹进行预测
        all_tracks = self.confirmed_tracks + self.tentative_tracks
        for track in all_tracks:
            track.predict()

        # 分层数据关联
        matched_indices, unmatched_track_indices, unmatched_det_indices = \
            self._hierarchical_association(all_tracks, current_detections)

        # 更新已匹配的轨迹
        for track_idx, det_idx in matched_indices:
            track = all_tracks[track_idx]
            det_info = current_detections[det_idx]

            track.update(
                det_info['det'],
                det_info['raw_feature'],
                det_info['reduced_feature'],
                self.frame_count
            )

            # 确认试探性轨迹
            if track.state == TrackState.New and track.hits >= self.min_hits:
                track.activate(self.frame_count)
                # 从试探性列表移动到已确认列表
                if track in self.tentative_tracks:
                    self.tentative_tracks.remove(track)
                    self.confirmed_tracks.append(track)

            # 重新激活丢失的轨迹
            if track.state == TrackState.Lost:
                track.state = TrackState.Tracked
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                    self.confirmed_tracks.append(track)

        # 处理未匹配的轨迹
        for track_idx in unmatched_track_indices:
            track = all_tracks[track_idx]

            if track.state == TrackState.Tracked:
                if track.time_since_update > self.max_age:
                    track.mark_removed()
                    if track in self.confirmed_tracks:
                        self.confirmed_tracks.remove(track)
                    self.removed_tracks.append(track)
                else:
                    track.mark_lost()
                    if track in self.confirmed_tracks:
                        self.confirmed_tracks.remove(track)
                        self.lost_tracks.append(track)
            elif track.state == TrackState.New:
                # 移除未被确认的试探性轨迹
                if track.time_since_update > self.min_hits:
                    track.mark_removed()
                    if track in self.tentative_tracks:
                        self.tentative_tracks.remove(track)
                    self.removed_tracks.append(track)

        # 为未匹配的检测初始化新的轨迹
        for det_idx in unmatched_det_indices:
            det_info = current_detections[det_idx]

            # 仅当置信度足够高时才初始化
            if det_info['det'][4] >= self.det_thresh:
                new_track = DFMTracklet(
                    det_info['det'],
                    det_info['raw_feature'],
                    det_info['reduced_feature'],
                    self.ukf_template,
                    self.frame_count
                )
                self.tentative_tracks.append(new_track)

        # 清理已移除的轨迹
        self.removed_tracks.clear()

        # 更新丢失的轨迹
        lost_to_remove = []
        for track in self.lost_tracks:
            track.predict()  # 对丢失的轨迹继续进行预测
            if track.time_since_update > self.max_age:
                track.mark_removed()
                lost_to_remove.append(track)

        for track in lost_to_remove:
            self.lost_tracks.remove(track)
            self.removed_tracks.append(track)

        # 为已确认的轨迹生成输出
        outputs = []
        for track in self.confirmed_tracks:
            if track.state == TrackState.Tracked:
                output = np.concatenate([
                    track.tlbr,
                    [track.track_id, track.conf, track.cls, track.det_ind]
                ])
                outputs.append(output)

        # 存储活动轨迹以供可视化
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


# 示例用法和测试
if __name__ == "__main__":
    # 初始化跟踪器
    tracker = DFMTrack(
        original_feature_dim=512,
        reduced_feature_dim=64,
        ukf_q_std=0.01,
        ukf_r_std=0.1,
        mahalanobis_gate_thresh=0.95,
        cosine_dist_thresh=0.4,
        det_thresh=0.3,
        max_age=30,
        min_hits=3
    )

