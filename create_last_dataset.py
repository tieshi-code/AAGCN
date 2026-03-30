#!/usr/bin/env python3
"""
创建last数据集
基于五个测试整合标签（已过滤无视频记录），为AlphaPose视频文件生成五折交叉验证数据集
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def normalize_dp_name(dp_raw: str) -> str:
    """
    将Data Point名称规范化，解决以下问题：
    1) 路径中含有空格: "Data point 10" -> "Data point10"
    2) 路径中含有后缀: "Data point10.1" -> "Data point10"
    3) 统一大小写与空格
    """
    if dp_raw is None:
        return None
    dp = dp_raw.strip()
    # 去掉中间多余空格，例如 "Data point 10" -> "Data point10"
    dp = dp.replace("Data point ", "Data point")
    # 去掉小数后缀 (.1/.2/.3/.4 等)
    if "." in dp:
        dp = dp.split(".")[0]
    return dp


def load_excel_labels(excel_file):
    """加载Excel文件中的标签信息"""
    print("加载Excel标签文件...")
    df = pd.read_excel(excel_file)

    # 创建LK-Data point到标签的映射
    label_map = {}
    for _, row in df.iterrows():
        lk = str(row['LK']).strip()
        dp = normalize_dp_name(str(row['Data_Point']))
        key = (lk, dp)
        label_map[key] = int(row['整合标签'])

    print(f"加载了 {len(label_map)} 个标签映射")
    return label_map


def find_video_files(alphapose_dirs):
    """查找所有AlphaPose视频文件"""
    print("查找AlphaPose视频文件...")
    video_files = []

    for base_dir in alphapose_dirs:
        if not base_dir.exists():
            print(f"警告: 目录不存在 {base_dir}")
            continue

        # 遍历所有子目录查找mp4文件
        for mp4_file in base_dir.rglob('**/AlphaPose_*.mp4'):
            video_files.append(mp4_file)

    print(f"找到 {len(video_files)} 个视频文件")
    return video_files


def parse_video_info(video_path):
    """从视频路径解析LK和Data point信息"""
    path_parts = video_path.parts

    lk = None
    data_point = None

    # 从路径中提取LK和Data point
    for part in path_parts:
        if part.startswith('LK'):
            lk = part
        elif part.startswith('Data point'):
            data_point = part

    return lk, normalize_dp_name(data_point)


def find_json_for_video(video_path):
    """为视频文件找到对应的JSON文件"""
    # JSON文件通常在同一目录下
    json_dir = video_path.parent
    json_files = list(json_dir.glob('**/alphapose-results.json'))

    if json_files:
        return json_files[0]  # 返回第一个找到的JSON文件
    return None


def process_video_data(json_file, label):
    """处理单个视频的数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            alphapose_data = json.load(f)

        # AlphaPose参数
        MAX_FRAME = 300
        NUM_JOINTS = 17
        C = 3  # x, y, confidence

        # 初始化数据数组
        data_array = np.zeros((C, MAX_FRAME, NUM_JOINTS, 1), dtype=np.float32)

        if isinstance(alphapose_data, list) and alphapose_data:
            # 收集所有帧的数据
            frames_dict = {}
            for item in alphapose_data:
                frame_id = None
                if "image_id" in item:
                    img_id = item["image_id"]
                    try:
                        # 尝试从image_id中提取帧号
                        frame_str = img_id.split('_')[-1].split('.')[0]
                        frame_id = int(frame_str)
                    except:
                        pass
                elif "frame_index" in item:
                    frame_id = int(item["frame_index"])

                if frame_id is not None and "keypoints" in item:
                    frames_dict[frame_id] = item["keypoints"]

            # 填充数据
            if frames_dict:
                max_frame = max(frames_dict.keys())
                original_frames = sorted(frames_dict.keys())

                # 如果超过MAX_FRAME则均匀采样
                if len(original_frames) > MAX_FRAME:
                    indices = np.linspace(0, len(original_frames) - 1, MAX_FRAME, dtype=int)
                    selected_frames = [original_frames[i] for i in indices]
                else:
                    selected_frames = original_frames

                for t, frame_id in enumerate(selected_frames[:MAX_FRAME]):
                    if frame_id in frames_dict:
                        keypoints = frames_dict[frame_id]

                        # 提取x, y, score (AlphaPose格式: [x1,y1,s1,x2,y2,s2,...])
                        xs = []
                        ys = []
                        scores = []

                        for i in range(0, len(keypoints), 3):
                            if i + 2 < len(keypoints):
                                xs.append(keypoints[i])      # x坐标
                                ys.append(keypoints[i + 1])  # y坐标
                                scores.append(keypoints[i + 2])  # 置信度

                        # 确保有17个关键点
                        while len(xs) < NUM_JOINTS:
                            xs.append(0.0)
                            ys.append(0.0)
                            scores.append(0.0)

                        # 填充到数组
                        data_array[0, t, :, 0] = xs[:NUM_JOINTS]
                        data_array[1, t, :, 0] = ys[:NUM_JOINTS]
                        data_array[2, t, :, 0] = scores[:NUM_JOINTS]

        return data_array, label

    except Exception as e:
        print(f"处理视频数据失败 {json_file}: {e}")
        return None, None


def create_last_dataset():
    """创建last数据集"""
    print("=== 创建last数据集 ===\n")

    # 文件路径
    excel_file = '/home/weiping/st-gcn-master/st-gcn-master/五个测试整合标签详细表格.xlsx'
    alphapose_dirs = [
        Path('/home/weiping/AlphaPose-master/视频数据/单直走/output'),
        Path('/home/weiping/AlphaPose-master/视频数据/单tug/output')
    ]
    output_dir = '/home/weiping/st-gcn-master/st-gcn-master/data/last'

    # 1. 加载标签
    label_map = load_excel_labels(excel_file)

    # 2. 查找视频文件
    video_files = find_video_files(alphapose_dirs)

    # 3. 处理视频数据
    print("\n处理视频数据...")
    data_list = []
    label_list = []
    sample_names = []
    processed_count = 0

    for video_path in video_files:
        # 解析视频信息
        lk, data_point = parse_video_info(video_path)

        if lk is None or data_point is None:
            print(f"警告: 无法解析视频路径 {video_path}")
            continue

        # 获取标签
        key = (lk, data_point)
        if key not in label_map:
            # 不输出警告，因为已经在Excel中过滤掉了没有视频的记录
            continue

        label = label_map[key]

        # 找到对应的JSON文件
        json_file = find_json_for_video(video_path)
        if json_file is None:
            print(f"警告: 找不到JSON文件 {video_path}")
            continue

        # 处理视频数据
        data, processed_label = process_video_data(json_file, label)
        if data is not None:
            data_list.append(data)
            label_list.append(processed_label)
            sample_names.append(f"{lk}_{data_point}_{processed_count}")

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"已处理 {processed_count}/{len(video_files)} 个视频")

    print(f"\n成功处理 {len(data_list)} 个视频文件")

    if not data_list:
        print("没有成功处理任何视频文件")
        return

    # 4. 转换为numpy数组
    all_data = np.stack(data_list, axis=0)
    all_labels = np.array(label_list)

    print(f"数据形状: {all_data.shape}")
    print(f"标签分布: {dict(sorted(pd.Series(all_labels).value_counts().sort_index().items()))}")

    # 5. 保存完整数据集
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整数据
    np.save(os.path.join(output_dir, 'full_data.npy'), all_data)
    with open(os.path.join(output_dir, 'full_labels.pkl'), 'wb') as f:
        pickle.dump((all_labels.tolist(), sample_names), f)

    print(f"完整数据集已保存到: {output_dir}")

    # 6. 生成五折交叉验证（分层抽样，确保每折标签分布均匀）
    print("\n生成五折交叉验证（分层抽样）...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(all_labels))

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, all_labels)):
        print(f"\nFold {fold_idx + 1}:")

        # 创建fold目录
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        # 分割数据
        train_data = all_data[train_indices]
        train_labels_fold = all_labels[train_indices]
        train_names = [sample_names[i] for i in train_indices]

        val_data = all_data[val_indices]
        val_labels_fold = all_labels[val_indices]
        val_names = [sample_names[i] for i in val_indices]

        # 统计分布
        train_dist = pd.Series(train_labels_fold).value_counts().sort_index()
        val_dist = pd.Series(val_labels_fold).value_counts().sort_index()

        print(f"  训练集: {len(train_labels_fold)} 样本")
        print(f"    标签分布: {dict(train_dist)}")
        print(f"  验证集: {len(val_labels_fold)} 样本")
        print(f"    标签分布: {dict(val_dist)}")

        # 保存训练数据
        np.save(os.path.join(fold_dir, 'train_data.npy'), train_data)
        with open(os.path.join(fold_dir, 'train_label.pkl'), 'wb') as f:
            pickle.dump((train_labels_fold.tolist(), train_names), f)

        # 保存验证数据
        np.save(os.path.join(fold_dir, 'val_data.npy'), val_data)
        with open(os.path.join(fold_dir, 'val_label.pkl'), 'wb') as f:
            pickle.dump((val_labels_fold.tolist(), val_names), f)

        print(f"  保存到: {fold_dir}")

    print("\n✓ 五折交叉验证数据生成完成!")
    print(f"输出目录: {output_dir}")

    # 生成配置文件
    print("\n生成配置文件...")
    config_dir = '/home/weiping/st-gcn-master/st-gcn-master/config/last'
    os.makedirs(config_dir, exist_ok=True)

    # 配置模板（基于备份中整合标签的成功配置）
    config_template = '''# command line: main.py recognition -c config/last/train_fold{fold}.yaml

T_max: 80
base_lr: 0.0003
batch_size: 8
config: "config/last/train_fold{fold}.yaml"
debug: false
device:
- 3
early_stop_patience: null
eta_min: 1.0e-06
eval_interval: 5
feeder: feeder.feeder_kinetics.Feeder_kinetics
gamma: 0.1
ignore_weights: []
log_interval: 100
model: net.st_gcn.Model
model_args:
    edge_importance_weighting: true
    graph_args:
        layout: alphapose
        strategy: spatial
    in_channels: 3
    num_class: 5
nesterov: true
num_epoch: 80
num_worker: 4
optimizer: AdamW
pavi_log: false
phase: train
print_log: true
save_interval: 10
save_log: true
save_result: false
scheduler: CosineAnnealingLR
scheduler_factor: 0.3
scheduler_min_lr: 1.0e-07
scheduler_mode: min
scheduler_patience: 7
scheduler_threshold: 0.0001
scheduler_verbose: true
show_topk:
- 1
- 5
start_epoch: 0
step: []
step_size: 10
test_batch_size: 8
test_feeder_args:
    data_path: "./data/last/fold_{fold}"
    ignore_empty_sample: true
    label_path: "./data/last/fold_{fold}/val_label.pkl"
    lower_body_only: true
    num_person_in: 1
    num_person_out: 1
    upper_body_scale: 0.0
train_feeder_args:
    data_path: "./data/last/fold_{fold}"
    ignore_empty_sample: true
    label_path: "./data/last/fold_{fold}/train_label.pkl"
    lower_body_only: true
    num_person_in: 1
    num_person_out: 1
    random_choose: true
    random_move: true
    random_shift: true
    upper_body_scale: 0.0
    window_size: 300
use_class_weight: false
use_gpu: true
warmup: true
warmup_epochs: 5
warmup_start_lr: 1.0e-06
weight_decay: 0.01
weights: null
work_dir: "./work_dir/recognition/last/ST_GCN_fold{fold}"
'''

    for fold_idx in range(1, 6):
        config_content = config_template.format(fold=fold_idx)
        config_file = os.path.join(config_dir, f'train_fold{fold_idx}.yaml')
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"创建配置文件: {config_file}")

    print("\n✓ 配置文件生成完成!")
    print(f"配置文件目录: {config_dir}")
    print("\n可以使用以下命令开始训练:")
    print("python3 main.py recognition -c config/last/train_fold1.yaml")


if __name__ == "__main__":
    create_last_dataset()
