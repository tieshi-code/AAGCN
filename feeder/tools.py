import numpy as np
import random


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
            t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def alphapose_match(data_numpy):
    # AlphaPose 與 OpenPose 在此匹配邏輯上可共用（僅依據關節座標與分數，不依賴關節數目常數）
    return openpose_match(data_numpy)

def _get_flip_pairs(layout, num_joint):
    if layout == 'alphapose' or num_joint == 17:
        return [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                (11, 12), (13, 14), (15, 16)]
    if layout == 'ntu-rgb+d' or num_joint == 25:
        return [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
                (21, 22), (23, 24)]
    # default openpose 18 joints
    return [(2, 5), (3, 6), (4, 7), (9, 12),
            (10, 13), (11, 14), (15, 16)]


def random_horizontal_flip(data_numpy, layout='alphapose', amputee_type='unilateral_lower'):
    """
    左右镜像翻转：谨慎使用
    只对双侧/上肢截肢或无截肢视频开启，单侧下肢截肢风险高，概率压低
    """
    C, T, V, M = data_numpy.shape
    if C < 2:
        return data_numpy
    
    # 单侧下肢截肢时不使用镜像翻转
    if amputee_type == 'unilateral_lower':
        return data_numpy
    
    data_flipped = data_numpy.copy()
    data_flipped[0] = -data_flipped[0]
    pairs = _get_flip_pairs(layout, V)
    for i, j in pairs:
        if i < V and j < V:
            tmp = data_flipped[:, :, i, :].copy()
            data_flipped[:, :, i, :] = data_flipped[:, :, j, :]
            data_flipped[:, :, j, :] = tmp
    return data_flipped


def random_temporal_crop(data_numpy, min_ratio=0.9):
    C, T, V, M = data_numpy.shape
    if not (0 < min_ratio < 1.0):
        return data_numpy
    keep_len = max(1, int(T * np.random.uniform(min_ratio, 1.0)))
    if keep_len >= T:
        return data_numpy
    begin = np.random.randint(0, T - keep_len + 1)
    cropped = data_numpy[:, begin:begin + keep_len, :, :]
    out = np.zeros_like(data_numpy)
    out[:, :keep_len, :, :] = cropped
    return out


def random_gaussian_jitter(data_numpy, sigma=0.02):
    if sigma <= 0:
        return data_numpy
    jitter = np.random.normal(0, sigma, size=data_numpy[0:2].shape)
    data_noised = data_numpy.copy()
    data_noised[0:2] = data_noised[0:2] + jitter
    return data_noised


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


# ========== 数据增强方法 ==========

def random_rotation(data_numpy, angle_range=(-30, 30)):
    """
    随机旋转：对整个序列在水平面旋转 ±15°–30°
    模拟不同拍摄角度，提升 3–8% 准确率
    """
    if angle_range[0] == 0 and angle_range[1] == 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    if C < 2:
        return data_numpy
    
    angle = np.random.uniform(angle_range[0], angle_range[1]) * np.pi / 180.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    data_rotated = data_numpy.copy()
    for t in range(T):
        xy = data_numpy[0:2, t, :, :].reshape(2, -1)
        rotated_xy = np.dot(rotation_matrix, xy)
        data_rotated[0:2, t, :, :] = rotated_xy.reshape(2, V, M)
    
    return data_rotated


def random_scaling(data_numpy, scale_min=0.95, scale_max=1.05):
    """
    随机缩放：整体尺度变化
    参数范围: 0.95 ~ 1.05，模拟身高/距离差异
    """
    if scale_min >= scale_max or scale_min <= 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    if C < 2:
        return data_numpy
    
    scale = np.random.uniform(scale_min, scale_max)
    
    data_scaled = data_numpy.copy()
    data_scaled[0:2] = data_scaled[0:2] * scale
    
    return data_scaled


def random_translation(data_numpy, translate_range=(-0.2, 0.2)):
    """
    随机平移：x/y/z 轴小范围偏移 ±0.1–0.2
    增加位置鲁棒性
    """
    if translate_range[0] == 0 and translate_range[1] == 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    if C < 2:
        return data_numpy
    
    translate_x = np.random.uniform(translate_range[0], translate_range[1])
    translate_y = np.random.uniform(translate_range[0], translate_range[1])
    
    data_translated = data_numpy.copy()
    data_translated[0] = data_translated[0] + translate_x
    data_translated[1] = data_translated[1] + translate_y
    
    return data_translated


def temporal_jitter(data_numpy, jitter_ratio=0.15):
    """
    时间抖动：随机加速/减速序列 ±10–20%（插值重采样）
    模拟速度变化
    """
    if jitter_ratio <= 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    
    # 随机选择加速或减速
    speed_factor = np.random.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)
    new_length = int(T * speed_factor)
    new_length = max(1, min(new_length, T * 2))  # 限制范围
    
    # 使用线性插值重采样
    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, new_length)
    
    data_jittered = np.zeros((C, new_length, V, M), dtype=data_numpy.dtype)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                data_jittered[c, :, v, m] = np.interp(
                    new_indices, old_indices, data_numpy[c, :, v, m])
    
    # 如果新长度小于原长度，填充；如果大于，裁剪
    if new_length < T:
        data_out = np.zeros_like(data_numpy)
        data_out[:, :new_length, :, :] = data_jittered
        return data_out
    elif new_length > T:
        return data_jittered[:, :T, :, :]
    else:
        return data_jittered


def joint_gaussian_noise(data_numpy, sigma_min=0.025, sigma_max=0.04):
    """
    高斯噪声：对关节坐标加高斯噪声
    参数范围: std 0.025 ~ 0.04，模拟 AlphaPose 检测噪声（假肢常有遮挡/误差）
    """
    if sigma_max <= 0 or sigma_min < 0:
        return data_numpy
    
    sigma = np.random.uniform(sigma_min, sigma_max)
    C, T, V, M = data_numpy.shape
    if C < 2:
        return data_numpy
    
    noise = np.random.normal(0, sigma, size=(2, T, V, M))
    data_noised = data_numpy.copy()
    data_noised[0:2] = data_noised[0:2] + noise
    
    return data_noised


def temporal_warp(data_numpy, ratio_min=0.88, ratio_max=1.12):
    """
    时间 warp：模拟患者自然速度波动
    参数范围: 0.88 ~ 1.12 倍速，范围控制小
    """
    if ratio_min >= ratio_max or ratio_min <= 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    ratio = np.random.uniform(ratio_min, ratio_max)
    new_length = int(T * ratio)
    new_length = max(1, min(new_length, int(T * 1.5)))  # 限制最大长度
    
    # 使用线性插值重采样
    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, new_length)
    
    data_warped = np.zeros((C, new_length, V, M), dtype=data_numpy.dtype)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                data_warped[c, :, v, m] = np.interp(
                    new_indices, old_indices, data_numpy[c, :, v, m])
    
    # 如果长度变化，需要调整回原始长度
    if new_length < T:
        data_out = np.zeros_like(data_numpy)
        data_out[:, :new_length, :, :] = data_warped
        return data_out
    elif new_length > T:
        return data_warped[:, :T, :, :]
    else:
        return data_warped


def temporal_shift(data_numpy, shift_max=15):
    """
    时间移位：处理视频起始帧不一致
    参数范围: ±15 帧
    """
    if shift_max <= 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    shift = np.random.randint(-shift_max, shift_max + 1)
    
    if shift == 0:
        return data_numpy
    
    data_shifted = np.zeros_like(data_numpy)
    if shift > 0:
        # 向右移动（用尾部补前面）
        data_shifted[:, shift:, :, :] = data_numpy[:, :-shift, :, :]
        data_shifted[:, :shift, :, :] = data_numpy[:, -shift:, :, :]
    else:
        # 向左移动（用前面补尾部）
        data_shifted[:, :shift, :, :] = data_numpy[:, -shift:, :, :]
        data_shifted[:, shift:, :, :] = data_numpy[:, :-shift, :, :]
    
    return data_shifted


def joint_random_mask(data_numpy, mask_ratio_min=0.12, mask_ratio_max=0.18, 
                      amputee_type='unilateral_lower'):
    """
    关节随机 mask：模拟部分遮挡，增强鲁棒性
    参数范围: 12~18% 关节，偏向非截肢侧
    amputee_type: 'unilateral_lower'(默认单侧下肢), 'bilateral', 'upper', 'none'
    """
    if mask_ratio_max <= 0 or mask_ratio_min < 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    mask_ratio = np.random.uniform(mask_ratio_min, mask_ratio_max)
    
    # 创建随机 mask (T, V, M)
    mask = np.random.rand(T, V, M) < mask_ratio
    
    # 单侧下肢截肢时，降低右腿 mask 概率
    if amputee_type == 'unilateral_lower':
        # AlphaPose 17 joints: 右腿关节 (12, 14, 16) = RHip, RKnee, RAnkle
        right_leg_joints = [12, 14, 16]
        for j in right_leg_joints:
            if j < V:
                mask[:, j, :] = mask[:, j, :] * (np.random.rand(T, M) < 0.4)  # 大幅降低概率
    
    # 应用 mask：将 masked 位置设为0
    data_masked = data_numpy.copy()
    for c in range(C):
        data_masked[c, mask] = 0.0
    
    return data_masked


def drop_joint(data_numpy, drop_ratio=0.2):
    """
    骨架遮挡/丢弃：随机丢弃 10–30% 关节
    防止模型过度依赖特定关节
    """
    if drop_ratio <= 0:
        return data_numpy
    
    C, T, V, M = data_numpy.shape
    num_drop = max(1, int(V * drop_ratio))
    
    drop_indices = np.random.choice(V, num_drop, replace=False)
    data_dropped = data_numpy.copy()
    data_dropped[:, :, drop_indices, :] = 0
    
    return data_dropped


def alpha_tug_augment_comprehensive(data_numpy, amputee_type='unilateral_lower', 
                                     enable_cross_temporal=True, enable_rotation=True,
                                     enable_scaling=True, enable_noise=True,
                                     enable_temporal_warp=True, enable_temporal_shift=True,
                                     enable_joint_mask=True, enable_flip=True):
    """
    AlphaPose TUG 数据增强综合方案
    
    增强项、开启概率、参数范围、目的:
    - 交叉抽帧: 0.65, 偶/奇索引拆分，增加时序相位多样性
    - 随机旋转: 0.65, ±20°（绕 Z 轴），模拟不同摄像机角度
    - 随机缩放: 0.65, 0.95 ~ 1.05，模拟身高/距离差异
    - 高斯噪声: 0.70, std 0.025 ~ 0.04，模拟 AlphaPose 检测噪声
    - 时间 warp: 0.50, 0.88 ~ 1.12 倍速，模拟患者自然速度波动
    - 时间移位: 0.45, ±15 帧，处理视频起始帧不一致
    - 关节随机 mask: 0.40, 12~18% 关节，偏向非截肢侧，模拟部分遮挡
    - 左右镜像翻转: 0.25, 只对双侧/上肢截肢或无截肢视频开启
    
    参数:
        data_numpy: (C, T, V, M) numpy array
        amputee_type: 'unilateral_lower'(默认), 'bilateral', 'upper', 'none'
    """
    augmented = data_numpy.copy()
    
    # 1. 交叉抽帧 (30→2×15fps) - 概率 0.65
    if enable_cross_temporal and np.random.rand() < 0.65:
        augmented = cross_temporal_sampling_enhanced(augmented, method='even_odd_split')
    
    # 2. 随机旋转 - 概率 0.65
    if enable_rotation and np.random.rand() < 0.65:
        augmented = random_rotation(augmented, angle_range=(-20, 20))
    
    # 3. 随机缩放 - 概率 0.65
    if enable_scaling and np.random.rand() < 0.65:
        augmented = random_scaling(augmented, scale_min=0.95, scale_max=1.05)
    
    # 4. 高斯噪声 - 概率 0.70
    if enable_noise and np.random.rand() < 0.70:
        augmented = joint_gaussian_noise(augmented, sigma_min=0.025, sigma_max=0.04)
    
    # 5. 时间 warp - 概率 0.50
    if enable_temporal_warp and np.random.rand() < 0.50:
        augmented = temporal_warp(augmented, ratio_min=0.88, ratio_max=1.12)
    
    # 6. 时间移位 - 概率 0.45
    if enable_temporal_shift and np.random.rand() < 0.45:
        augmented = temporal_shift(augmented, shift_max=15)
    
    # 7. 关节随机 mask - 概率 0.40
    if enable_joint_mask and np.random.rand() < 0.40:
        augmented = joint_random_mask(augmented, 
                                      mask_ratio_min=0.12, 
                                      mask_ratio_max=0.18,
                                      amputee_type=amputee_type)
    
    # 8. 左右镜像翻转 - 概率 0.25（谨慎使用）
    if enable_flip and np.random.rand() < 0.25:
        augmented = random_horizontal_flip(augmented, layout='alphapose', 
                                          amputee_type=amputee_type)
    
    return augmented


def skeleton_mixup(data_numpy1, data_numpy2, label1, label2, alpha=0.2):
    """
    混合增强（Mixup for skeleton）：在序列间线性插值
    进一步提升 2–5%（最新论文常见）
    """
    if alpha <= 0:
        return data_numpy1, label1
    
    lam = np.random.beta(alpha, alpha)
    
    # 两个序列长度取 min(T1,T2)
    C, T1, V, M = data_numpy1.shape
    _, T2, _, _ = data_numpy2.shape
    T = min(T1, T2)
    
    mixed_data = lam * data_numpy1[:, :T, :, :] + (1 - lam) * data_numpy2[:, :T, :, :]
    mixed_label = lam * label1 + (1 - lam) * label2
    
    # 如果T1 > T，需要填充
    if T1 > T:
        data_out = np.zeros_like(data_numpy1)
        data_out[:, :T, :, :] = mixed_data
        return data_out, mixed_label
    
    return mixed_data, mixed_label


def cross_temporal_sampling_enhanced(data_numpy, method='even_odd_split'):
    """
    交叉抽帧（增强版）：偶/奇索引拆分，padding 补齐
    30fps→2×15fps，增加时序相位多样性，TUG 慢动作友好
    返回：单个增强样本（随机选择偶或奇序列）
    """
    C, T, V, M = data_numpy.shape
    
    if T <= 1:
        return data_numpy
    
    if method == 'even_odd_split':
        # 偶/奇索引拆分
        seq_even = data_numpy[:, 0:T:2, :, :]   # 偶数索引
        seq_odd = data_numpy[:, 1:T:2, :, :]    # 奇数索引
        
        # 随机选择其中一个
        if np.random.rand() < 0.5:
            selected_seq = seq_even
            target_len = seq_odd.shape[1] if seq_odd.shape[1] > seq_even.shape[1] else seq_even.shape[1]
        else:
            selected_seq = seq_odd
            target_len = seq_even.shape[1] if seq_even.shape[1] > seq_odd.shape[1] else seq_odd.shape[1]
        
        # padding 补齐到目标长度
        current_len = selected_seq.shape[1]
        if current_len < target_len:
            # 用最后一帧填充
            pad_frames = target_len - current_len
            last_frame = selected_seq[:, -1:, :, :]
            pad = np.repeat(last_frame, pad_frames, axis=1)
            selected_seq = np.concatenate([selected_seq, pad], axis=1)
        elif current_len > target_len:
            selected_seq = selected_seq[:, :target_len, :, :]
        
        # 插值回原始长度
        if selected_seq.shape[1] != T:
            old_indices = np.linspace(0, selected_seq.shape[1] - 1, selected_seq.shape[1])
            new_indices = np.linspace(0, selected_seq.shape[1] - 1, T)
            
            data_out = np.zeros_like(data_numpy)
            for c in range(C):
                for v in range(V):
                    for m in range(M):
                        data_out[c, :, v, m] = np.interp(
                            new_indices, old_indices, selected_seq[c, :, v, m])
            return data_out
        
        return selected_seq
    
    # 默认使用原始方法
    return cross_temporal_sampling(data_numpy, downsample_ratio=0.5, method='linear')


def cross_temporal_sampling(data_numpy, downsample_ratio=0.5, method='linear'):
    """
    交叉抽帧（Cross-temporal Sampling）数据增强：
    1. 先抽帧（降采样）：从 T 帧中按比例抽取部分帧
    2. 后拉伸（插值）：将抽取的帧插值回原始长度 T
    
    参数:
        data_numpy: 输入数据 (C, T, V, M)
        downsample_ratio: 降采样比例，默认0.5（抽取一半帧）
        method: 插值方法，'linear' 或 'cubic'
    
    返回:
        data_numpy: 增强后的数据 (C, T, V, M)
    
    示例:
        400帧 -> 抽取200帧 -> 插值回400帧
        生成不同的时间视角，将172个样本变成344个
    """
    C, T, V, M = data_numpy.shape
    
    if T <= 1 or downsample_ratio >= 1.0 or downsample_ratio <= 0:
        return data_numpy
    
    # 1. 先抽帧（降采样）：随机抽取 downsample_ratio 比例的帧
    num_samples = max(1, int(T * downsample_ratio))
    
    # 随机选择要保留的帧索引（避免总是从开头或结尾）
    if num_samples == T:
        selected_indices = np.arange(T)
    else:
        # 选择的帧均匀分布在序列中
        step = T / num_samples
        base_indices = np.linspace(0, T - 1, num_samples).astype(int)
        # 添加小范围随机抖动，避免完全均匀
        jitter = np.random.randint(-max(1, int(step * 0.2)), max(1, int(step * 0.2)) + 1, size=num_samples)
        selected_indices = np.clip(base_indices + jitter, 0, T - 1)
        selected_indices = np.unique(selected_indices)  # 去重并排序
        if len(selected_indices) == 0:
            selected_indices = np.array([T // 2])  # 至少保留一帧
    
    # 获取降采样后的数据
    downsampled_data = data_numpy[:, selected_indices, :, :]  # (C, num_samples, V, M)
    num_samples = downsampled_data.shape[1]
    
    # 2. 后拉伸（插值）：将 num_samples 帧插值回 T 帧
    # 原始时间索引
    old_time_indices = np.linspace(0, T - 1, num_samples)
    # 目标时间索引
    new_time_indices = np.linspace(0, T - 1, T)
    
    # 使用线性插值或三次插值
    data_out = np.zeros_like(data_numpy)
    
    if method == 'linear' or num_samples < 4:
        # 线性插值
        for c in range(C):
            for v in range(V):
                for m in range(M):
                    data_out[c, :, v, m] = np.interp(
                        new_time_indices,
                        old_time_indices,
                        downsampled_data[c, :, v, m]
                    )
    else:
        # 三次样条插值（需要至少4个点）
        try:
            from scipy import interpolate
            has_scipy = True
        except ImportError:
            has_scipy = False
        
        if has_scipy and num_samples >= 4:
            # 使用三次样条插值
            for c in range(C):
                for v in range(V):
                    for m in range(M):
                        f = interpolate.interp1d(
                            old_time_indices,
                            downsampled_data[c, :, v, m],
                            kind='cubic',
                            fill_value='extrapolate'
                        )
                        data_out[c, :, v, m] = f(new_time_indices)
        else:
            # 没有 scipy 或点数不足，使用线性插值
            for c in range(C):
                for v in range(V):
                    for m in range(M):
                        data_out[c, :, v, m] = np.interp(
                            new_time_indices,
                            old_time_indices,
                            downsampled_data[c, :, v, m]
                        )
    
    # 保持置信度通道的一致性
    # 如果降采样时某帧的置信度为0，插值后应该也是0或接近0
    if C >= 3:
        # 对于置信度通道，使用最近邻或线性插值，但保持阈值
        score_downsampled = downsampled_data[2, :, :, :]
        # 创建掩码：原始帧中置信度为0的位置
        score_mask = (score_downsampled == 0)
        
        # 插值置信度
        for v in range(V):
            for m in range(M):
                data_out[2, :, v, m] = np.interp(
                    new_time_indices,
                    old_time_indices,
                    downsampled_data[2, :, v, m]
                )
        
        # 如果原始帧某关节置信度为0，插值后位置也应该设为0
        # 使用最近邻传播
        for i, old_idx in enumerate(selected_indices):
            if old_idx < T:
                # 找到插值后最接近的位置
                closest_new_idx = np.argmin(np.abs(new_time_indices - old_time_indices[i]))
                if score_mask[i, :, :].any():
                    # 传播置信度掩码（简单处理：在附近帧也设为0）
                    mask_v, mask_m = np.where(score_mask[i, :, :])
                    for v_idx in mask_v:
                        for m_idx in mask_m:
                            # 在插值后的帧中，对应关节置信度设为0
                            # 使用最近邻传播
                            radius = max(1, int(T / num_samples))
                            start_idx = max(0, closest_new_idx - radius)
                            end_idx = min(T, closest_new_idx + radius + 1)
                            # 只在接近原始位置的地方应用掩码
                            if abs(closest_new_idx - (T * i / num_samples)) < T / num_samples:
                                data_out[2, start_idx:end_idx, v_idx, m_idx] = 0
                                data_out[0, start_idx:end_idx, v_idx, m_idx] = 0
                                data_out[1, start_idx:end_idx, v_idx, m_idx] = 0
    
    return data_out