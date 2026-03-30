# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
try:
    from . import tools
except (ImportError, ValueError, SystemError):
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    # 获取当前文件所在目录
    try:
        feeder_dir = Path(__file__).parent
    except NameError:
        # 如果 __file__ 不存在，使用当前工作目录下的 feeder
        feeder_dir = Path.cwd() / "feeder"
    if str(feeder_dir) not in sys.path:
        sys.path.insert(0, str(feeder_dir))
    import tools


class Feeder_kinetics(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=1,
                 num_person_out=1,
                 debug=False,
                 cross_temporal_sampling=False,
                 downsample_ratio=0.5,
                 use_comprehensive_aug=False,
                 amputee_type='unilateral_lower',
                 lower_body_only=False,
                 upper_body_scale=0.0,
                 selected_joints=None):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample
        self.cross_temporal_sampling = cross_temporal_sampling
        self.downsample_ratio = downsample_ratio
        self.use_comprehensive_aug = use_comprehensive_aug
        self.amputee_type = amputee_type
        self.lower_body_only = lower_body_only
        self.upper_body_scale = upper_body_scale
        self.selected_joints = selected_joints

        self.load_data()

    def load_data(self):
        # 检查label_path是pkl格式还是json格式
        label_path = self.label_path
        
        # 如果是pkl文件，使用npy+pkl格式加载
        if label_path.endswith('.pkl'):
            self._load_data_npy_pkl()
            return
        
        # 否则使用原有的json格式加载
        # load file list
        self.sample_name = os.listdir(self.data_path)
        # 只保留JSON文件并按名称排序
        self.sample_name = sorted([name for name in self.sample_name if name.endswith('.json')])

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        # load label
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        
        # 处理缺失的键，只处理存在的样本
        valid_indices = []
        valid_sample_ids = []
        for i, id in enumerate(sample_id):
            if id in label_info:
                valid_indices.append(i)
                valid_sample_ids.append(id)
        
        if not valid_indices:
            raise ValueError(f"没有找到有效的标签信息。样本ID: {sample_id[:5]}...")
        
        # 只处理有效的样本
        self.sample_name = [self.sample_name[i] for i in valid_indices]
        # 将标签从 6-12 映射到 0-6（TUG 分数范围）
        labels_raw = np.array(
            [label_info[id]['label_index'] for id in valid_sample_ids])
        # 如果标签范围是 6-12，则减去 6 映射到 0-6
        if labels_raw.min() >= 6 and labels_raw.max() <= 12:
            self.label = labels_raw - 6
        else:
            self.label = labels_raw
        has_skeleton = np.array(
            [bool(label_info[id].get('has_skeleton', True)) for id in valid_sample_ids],
            dtype=bool)

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]        #只保留有骨架数据对映的标签

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3  #channel
        self.T = self.window_size if self.window_size > 0 else 300  #frame
        self.V = 17  #joint
        self.M = self.num_person_out  #person
    
    def _load_data_npy_pkl(self):
        """
        加载npy格式的数据和pkl格式的标签
        用于walk数据集等已经转换为npy格式的数据
        """
        label_path = self.label_path
        data_path = self.data_path
        
        # 加载标签文件（pkl格式）
        with open(label_path, 'rb') as f:
            labels_data = pickle.load(f)
        
        # pkl文件格式: (labels, sample_names)
        if isinstance(labels_data, tuple) and len(labels_data) == 2:
            labels_array, sample_names = labels_data
            self.label = np.array(labels_array, dtype=np.int64)
            self.sample_name = list(sample_names)
        else:
            raise ValueError(f"不支持的pkl格式: {type(labels_data)}")
        
        # 尝试加载截肢标签（如果存在）
        amputee_path = label_path.replace('_label.pkl', '_amputee.pkl')
        if os.path.exists(amputee_path):
            with open(amputee_path, 'rb') as f:
                amputee_data = pickle.load(f)
            if isinstance(amputee_data, tuple) and len(amputee_data) == 2:
                self.amputee_labels = list(amputee_data[0])
            else:
                self.amputee_labels = ['none'] * len(self.sample_name)
        else:
            self.amputee_labels = ['none'] * len(self.sample_name)
        
        # 加载数据文件（npy格式）
        # 根据label_path判断是train还是val
        if 'train_label.pkl' in label_path:
            data_file = os.path.join(data_path, 'train_data.npy')
        elif 'val_label.pkl' in label_path:
            data_file = os.path.join(data_path, 'val_data.npy')
        else:
            # 尝试自动检测
            train_file = os.path.join(data_path, 'train_data.npy')
            val_file = os.path.join(data_path, 'val_data.npy')
            if os.path.exists(train_file) and os.path.exists(val_file):
                raise ValueError("无法确定使用train_data.npy还是val_data.npy，请在label_path中明确指定")
            elif os.path.exists(train_file):
                data_file = train_file
            elif os.path.exists(val_file):
                data_file = val_file
            else:
                raise FileNotFoundError(f"找不到数据文件: {data_path}/train_data.npy 或 val_data.npy")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        # 加载npy数据
        self.data = np.load(data_file, mmap_mode='r')
        
        # 验证数据维度
        if len(self.data.shape) != 5:
            raise ValueError(f"数据维度错误，期望5维(N,C,T,V,M)，实际为{self.data.shape}")
        
        # 标签与样本数量对齐
        if len(self.label) != len(self.sample_name) or len(self.label) != self.data.shape[0]:
            min_len = min(len(self.label), len(self.sample_name), self.data.shape[0])
            self.label = self.label[:min_len]
            self.sample_name = self.sample_name[:min_len]
            self.data = self.data[:min_len]
        
        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        # 从数据中提取维度信息 (N, C, T, V, M)
        if self.data.shape:
            _, self.C, self.T, self.V, _ = self.data.shape
        else:
            self.C = 3  # channel
            self.T = 300  # frame
            self.V = 17  # joint
        self.C = 3  #channel
        self.T = 300#frame
        self.V = 17  #joint
        self.M = self.num_person_out  #person           

    def __len__(self):
        # 如果启用交叉抽帧，返回双倍长度（原始样本 + 增强样本）
        base_len = len(self.sample_name)
        if self.cross_temporal_sampling:
            return base_len * 2
        return base_len

    def __iter__(self):
        return self                 #允许迭代

    def __getitem__(self, index):

        # 如果启用交叉抽帧，前一半是原始样本，后一半是增强样本
        use_augmentation = False
        if self.cross_temporal_sampling:
            base_len = len(self.sample_name)
            if index >= base_len:
                # 使用增强样本
                use_augmentation = True
                index = index - base_len
        
        # 检查是否使用npy格式（通过检查self.label_path是否以.pkl结尾）
        if hasattr(self, 'data') and isinstance(self.data, np.ndarray):
            # 使用npy格式加载数据
            # output shape (C, T, V, M)
            data_numpy = self.data[index].copy()  # (C, T, V, M)
            
            # 处理维度 (C,T,V,M) 或 (N,C,T,V,M)
            if len(data_numpy.shape) == 4:
                # 已经是 (C, T, V, M)
                pass
            elif len(data_numpy.shape) == 5:
                # 如果是 (N, C, T, V, M)，取第一个
                data_numpy = data_numpy[0]
            else:
                raise ValueError(f"数据维度错误: {data_numpy.shape}")
            
            # 获取标签
            label = int(self.label[index])
        else:
            # 使用JSON格式加载数据（原有逻辑）
            # output shape (C, T, V, M)
            # get data
            sample_name = self.sample_name[index]
            sample_path = os.path.join(self.data_path, sample_name)    
            with open(sample_path, 'r') as f:
                video_info = json.load(f)                       #获取第i个样本数据      

            # fill data_numpy
            data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
            for frame_info in video_info['data']:          #遍历每一帧的数据
                frame_index = frame_info['frame_index']      #得到当前第几帧的索引信息
                # 只保留前 T 帧（frame_index < T），超过的帧跳过
                if frame_index >= self.T:
                    continue
                for m, skeleton_info in enumerate(frame_info["skeleton"]):#遍历当前帧的所有信息
                    if m >= self.num_person_in:
                        break
                    pose = skeleton_info['pose']          #关节坐标 
                    score = skeleton_info['score']
                    data_numpy[0, frame_index, :, m] = pose[0::2]
                    data_numpy[1, frame_index, :, m] = pose[1::2]
                    data_numpy[2, frame_index, :, m] = score

            # centralization
            data_numpy[0:2] = data_numpy[0:2] - 0.5  #归一化
            data_numpy[0][data_numpy[2] == 0] = 0     
            data_numpy[1][data_numpy[2] == 0] = 0     #置信度为0的位置坐标设为0

            # get & check label index
            label_raw = video_info.get('label_index', self.label[index])  # 如果JSON中没有，使用数组中的标签
            # 将标签从 6-12 映射到 0-6（TUG 分数范围）
            if label_raw >= 6 and label_raw <= 12:
                label = label_raw - 6
            else:
                label = label_raw
            # 使用标签数组中的标签（与数据一致）
            label = int(self.label[index])
        
        # 对npy格式也进行归一化处理
        if hasattr(self, 'data') and isinstance(self.data, np.ndarray):
            # 中心化处理（相对于第一个关键点的位置）
            if np.sum(data_numpy[0, :, 0, 0]) > 0:  # 如果第一个关键点存在
                center_x = np.mean(data_numpy[0, :, 0, 0][data_numpy[0, :, 0, 0] > 0])
                center_y = np.mean(data_numpy[1, :, 0, 0][data_numpy[1, :, 0, 0] > 0])
                data_numpy[0, :, :, :] -= center_x
                data_numpy[1, :, :, :] -= center_y

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        
        # 数据增强
        if use_augmentation:
            if self.use_comprehensive_aug:
                # 使用综合增强方案
                data_numpy = tools.alpha_tug_augment_comprehensive(
                    data_numpy,
                    amputee_type=self.amputee_type
                )
            elif self.cross_temporal_sampling:
                # 仅使用交叉抽帧增强
                data_numpy = tools.cross_temporal_sampling(
                    data_numpy, 
                    downsample_ratio=self.downsample_ratio,
                    method='linear'
                )

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1,2,0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # 处理关键点选择
        if self.selected_joints is not None:
            if hasattr(self, 'data'):
                # 对于.npy格式数据，应用关键点选择
                # 创建一个17关键点的完整数组，在选定位置保留数据，其他位置为0
                selected_data = np.zeros((self.C, self.T, 17, self.num_person_out))

                for joint_idx in self.selected_joints:
                    if joint_idx < data_numpy.shape[2] and joint_idx < 17:
                        selected_data[:, :, joint_idx, :] = data_numpy[:, :, joint_idx, :]

                data_numpy = selected_data
            else:
                # 对于JSON格式数据，应用关键点选择
                # 创建一个17关键点的完整数组，在选定位置填充数据，其他位置为0
                padded_data = np.zeros((self.C, self.T, 17, self.num_person_out))

                for i, joint_idx in enumerate(self.selected_joints):
                    if i < data_numpy.shape[2] and joint_idx < 17:
                        padded_data[:, :, joint_idx, :] = data_numpy[:, :, i, :]

                data_numpy = padded_data

            # 保持V维度为17（与图结构匹配）
            self.V = 17
        elif self.lower_body_only:
            # 仅当没有指定selected_joints时才应用lower_body_only
            upper_end = min(11, data_numpy.shape[2])
            if self.upper_body_scale <= 0:
                data_numpy[:, :, :upper_end, :] = 0
            else:
                data_numpy[:, :, :upper_end, :] *= float(self.upper_body_scale)

        # match poses between 2 frames
        if self.pose_matching:
            if hasattr(self, 'layout') and self.layout == 'alphapose':
                # 使用AlphaPose的姿态匹配
                data_numpy = tools.alphapose_match(data_numpy)
            else:
                # 使用OpenPose的姿态匹配
                data_numpy = tools.openpose_match(data_numpy)

        # 返回数据、标签和截肢标签（如果存在）
        if hasattr(self, 'amputee_labels'):
            amputee_type = self.amputee_labels[index]
            return data_numpy, label, amputee_type
        else:
            return data_numpy, label    

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
