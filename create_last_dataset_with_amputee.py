#!/usr/bin/env python3
"""
创建last数据集（包含截肢标签）
基于五个测试整合标签，为AlphaPose视频文件生成五折交叉验证数据集
同时包含截肢腿标签（左右截肢腿）
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def normalize_dp_name(dp_raw: str) -> str:
    """标准化Data Point名称"""
    if dp_raw is None:
        return None
    dp = str(dp_raw).strip()
    dp = dp.replace("Data point ", "Data point")
    if "." in dp:
        dp = dp.split(".")[0]
    return dp


def load_excel_labels(excel_file):
    """加载Excel文件中的标签信息和截肢标签"""
    print("加载Excel标签文件...")
    df = pd.read_excel(excel_file)

    # 创建LK-Data point到标签的映射
    label_map = {}
    amputee_map = {}  # 截肢标签映射
    
    for _, row in df.iterrows():
        lk = str(row['LK']).strip()
        dp = normalize_dp_name(str(row['Data_Point']))
        key = (lk, dp)
        label_map[key] = int(row['整合标签'])
        
        # 加载截肢标签
        amputee_label = str(row.get('左右截肢腿', '')).strip()
        # 转换为模型可用的格式
        if amputee_label == '右腿截肢':
            amputee_map[key] = 'right_leg'
        elif amputee_label == '左腿截肢':
            amputee_map[key] = 'left_leg'
        elif amputee_label == '左腿和右腿都截肢':
            amputee_map[key] = 'both_legs'
        else:
            amputee_map[key] = 'none'

    print(f"加载了 {len(label_map)} 个标签映射")
    print(f"加载了 {len(amputee_map)} 个截肢标签映射")
    return label_map, amputee_map


def find_video_files(alphapose_dirs):
    """查找所有AlphaPose视频文件"""
    print("查找AlphaPose视频文件...")
    video_files = []

    for base_dir in alphapose_dirs:
        if not base_dir.exists():
            print(f"警告: 目录不存在 {base_dir}")
            continue

        for mp4_file in base_dir.rglob('**/AlphaPose_*.mp4'):
            video_files.append(mp4_file)

    print(f"找到 {len(video_files)} 个视频文件")
    return video_files


def parse_video_info(video_path):
    """从视频路径解析LK和Data point信息"""
    path_parts = video_path.parts
    lk = None
    data_point = None

    for part in path_parts:
        if part.startswith('LK'):
            lk = part
        elif part.startswith('Data point'):
            data_point = part

    return lk, normalize_dp_name(data_point)


def find_json_for_video(video_path):
    """为视频文件找到对应的JSON文件"""
    json_dir = video_path.parent
    json_files = list(json_dir.glob('**/alphapose-results.json'))

    if json_files:
        return json_files[0]
    return None


def process_video_data(json_file, label):
    """处理单个视频的数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            alphapose_data = json.load(f)

        MAX_FRAME = 300
        NUM_JOINTS = 17
        C = 3

        data_array = np.zeros((C, MAX_FRAME, NUM_JOINTS, 1), dtype=np.float32)

        if isinstance(alphapose_data, list) and alphapose_data:
            frames_dict = {}
            for item in alphapose_data:
                frame_id = None
                if "image_id" in item:
                    img_id = item["image_id"]
                    try:
                        frame_str = img_id.split('_')[-1].split('.')[0]
                        frame_id = int(frame_str)
                    except:
                        pass
                elif "frame_index" in item:
                    frame_id = int(item["frame_index"])

                if frame_id is not None and "keypoints" in item:
                    frames_dict[frame_id] = item["keypoints"]

            if frames_dict:
                original_frames = sorted(frames_dict.keys())
                if len(original_frames) > MAX_FRAME:
                    indices = np.linspace(0, len(original_frames) - 1, MAX_FRAME, dtype=int)
                    selected_frames = [original_frames[i] for i in indices]
                else:
                    selected_frames = original_frames

                for t, frame_id in enumerate(selected_frames[:MAX_FRAME]):
                    if frame_id in frames_dict:
                        keypoints = frames_dict[frame_id]
                        xs, ys, scores = [], [], []

                        for i in range(0, len(keypoints), 3):
                            if i + 2 < len(keypoints):
                                xs.append(keypoints[i])
                                ys.append(keypoints[i + 1])
                                scores.append(keypoints[i + 2])

                        while len(xs) < NUM_JOINTS:
                            xs.append(0.0)
                            ys.append(0.0)
                            scores.append(0.0)

                        data_array[0, t, :, 0] = xs[:NUM_JOINTS]
                        data_array[1, t, :, 0] = ys[:NUM_JOINTS]
                        data_array[2, t, :, 0] = scores[:NUM_JOINTS]

        return data_array, label

    except Exception as e:
        print(f"处理视频数据失败 {json_file}: {e}")
        return None, None


def create_last_dataset_with_amputee():
    """创建包含截肢标签的last数据集"""
    print("=== 创建last数据集（含截肢标签）===\n")

    excel_file = '/home/weiping/st-gcn-master/st-gcn-master/五个测试整合标签详细表格.xlsx'
    alphapose_dirs = [
        Path('/home/weiping/AlphaPose-master/视频数据/单直走/output'),
        Path('/home/weiping/AlphaPose-master/视频数据/单tug/output')
    ]
    output_dir = '/home/weiping/st-gcn-master/st-gcn-master/data/last_amputee'

    # 1. 加载标签和截肢标签
    label_map, amputee_map = load_excel_labels(excel_file)

    # 2. 查找视频文件
    video_files = find_video_files(alphapose_dirs)

    # 3. 处理视频数据
    print("\n处理视频数据...")
    data_list = []
    label_list = []
    amputee_list = []  # 截肢标签列表
    sample_names = []
    processed_count = 0

    for video_path in video_files:
        lk, data_point = parse_video_info(video_path)

        if lk is None or data_point is None:
            continue

        key = (lk, data_point)
        if key not in label_map:
            continue

        label = label_map[key]
        amputee_type = amputee_map.get(key, 'none')

        json_file = find_json_for_video(video_path)
        if json_file is None:
            continue

        data, processed_label = process_video_data(json_file, label)
        if data is not None:
            data_list.append(data)
            label_list.append(processed_label)
            amputee_list.append(amputee_type)
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
    print(f"标签分布: {dict(sorted(Counter(all_labels).items()))}")
    print(f"截肢标签分布: {dict(Counter(amputee_list))}")

    # 5. 保存完整数据集
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'full_data.npy'), all_data)
    with open(os.path.join(output_dir, 'full_labels.pkl'), 'wb') as f:
        pickle.dump((all_labels.tolist(), sample_names), f)
    
    # 保存截肢标签
    with open(os.path.join(output_dir, 'full_amputee_labels.pkl'), 'wb') as f:
        pickle.dump((amputee_list, sample_names), f)

    print(f"完整数据集已保存到: {output_dir}")

    # 6. 生成五折交叉验证
    print("\n生成五折交叉验证...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(all_labels))

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, all_labels)):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]
        train_amputee = [amputee_list[i] for i in train_indices]
        train_names = [sample_names[i] for i in train_indices]

        val_data = all_data[val_indices]
        val_labels = all_labels[val_indices]
        val_amputee = [amputee_list[i] for i in val_indices]
        val_names = [sample_names[i] for i in val_indices]

        np.save(os.path.join(fold_dir, 'train_data.npy'), train_data)
        with open(os.path.join(fold_dir, 'train_label.pkl'), 'wb') as f:
            pickle.dump((train_labels.tolist(), train_names), f)
        with open(os.path.join(fold_dir, 'train_amputee.pkl'), 'wb') as f:
            pickle.dump((train_amputee, train_names), f)

        np.save(os.path.join(fold_dir, 'val_data.npy'), val_data)
        with open(os.path.join(fold_dir, 'val_label.pkl'), 'wb') as f:
            pickle.dump((val_labels.tolist(), val_names), f)
        with open(os.path.join(fold_dir, 'val_amputee.pkl'), 'wb') as f:
            pickle.dump((val_amputee, val_names), f)

        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        print(f"Fold {fold_idx + 1}: 训练 {len(train_labels)}, 验证 {len(val_labels)}")
        print(f"  训练分布: {dict(sorted(train_dist.items()))}")
        print(f"  验证分布: {dict(sorted(val_dist.items()))}")

    print("\n✓ 数据集生成完成!")


if __name__ == "__main__":
    create_last_dataset_with_amputee()
