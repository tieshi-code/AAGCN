#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def normalize_dp_name(dp_raw: str) -> str:
    if dp_raw is None:
        return None
    dp = dp_raw.strip()
    dp = dp.replace("Data point ", "Data point")
    if "." in dp:
        dp = dp.split(".")[0]
    return dp


def load_excel_labels(excel_file):
    print("Loading Excel label file...")
    df = pd.read_excel(excel_file)

    label_map = {}
    for _, row in df.iterrows():
        lk = str(row['LK']).strip()
        dp = normalize_dp_name(str(row['Data_Point']))
        key = (lk, dp)
        label_map[key] = int(row['整合标签'])

    print(f"Loaded {len(label_map)} label mappings")
    return label_map


def find_video_files(alphapose_dirs):
    print("Finding AlphaPose video files...")
    video_files = []

    for base_dir in alphapose_dirs:
        if not base_dir.exists():
            print(f"Warning: Directory does not exist {base_dir}")
            continue

        for mp4_file in base_dir.rglob('**/AlphaPose_*.mp4'):
            video_files.append(mp4_file)

    print(f"Found {len(video_files)} video files")
    return video_files


def parse_video_info(video_path):
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
    json_dir = video_path.parent
    json_files = list(json_dir.glob('**/alphapose-results.json'))

    if json_files:
        return json_files[0]
    return None


def process_video_data(json_file, label):
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
                max_frame = max(frames_dict.keys())
                original_frames = sorted(frames_dict.keys())

                if len(original_frames) > MAX_FRAME:
                    indices = np.linspace(0, len(original_frames) - 1, MAX_FRAME, dtype=int)
                    selected_frames = [original_frames[i] for i in indices]
                else:
                    selected_frames = original_frames

                for t, frame_id in enumerate(selected_frames[:MAX_FRAME]):
                    if frame_id in frames_dict:
                        keypoints = frames_dict[frame_id]

                        xs = []
                        ys = []
                        scores = []

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
        print(f"Failed to process video data {json_file}: {e}")
        return None, None


def create_last_dataset():
    print("=== Creating last dataset ===\n")

    excel_file = '/home/weiping/st-gcn-master/st-gcn-master/111.xlsx'
    alphapose_dirs = [
        Path(),
        Path()
    ]
    output_dir = '/home/weiping/st-gcn-master/st-gcn-master/data/last'

    label_map = load_excel_labels(excel_file)

    video_files = find_video_files(alphapose_dirs)

    print("\nProcessing video data...")
    data_list = []
    label_list = []
    sample_names = []
    processed_count = 0

    for video_path in video_files:
        lk, data_point = parse_video_info(video_path)

        if lk is None or data_point is None:
            print(f"Warning: Could not parse video path {video_path}")
            continue

        key = (lk, data_point)
        if key not in label_map:
            continue

        label = label_map[key]

        json_file = find_json_for_video(video_path)
        if json_file is None:
            print(f"Warning: Could not find JSON file for {video_path}")
            continue

        data, processed_label = process_video_data(json_file, label)
        if data is not None:
            data_list.append(data)
            label_list.append(processed_label)
            sample_names.append(f"{lk}_{data_point}_{processed_count}")

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(video_files)} videos")

    print(f"\nSuccessfully processed {len(data_list)} video files")

    if not data_list:
        print("Failed to process any video files")
        return

    all_data = np.stack(data_list, axis=0)
    all_labels = np.array(label_list)

    print(f"Data shape: {all_data.shape}")
    print(f"Label distribution: {dict(sorted(pd.Series(all_labels).value_counts().sort_index().items()))}")

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'full_data.npy'), all_data)
    with open(os.path.join(output_dir, 'full_labels.pkl'), 'wb') as f:
        pickle.dump((all_labels.tolist(), sample_names), f)

    print(f"Full dataset saved to: {output_dir}")

    print("\nGenerating 5-fold cross-validation (stratified)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(all_labels))

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, all_labels)):
        print(f"\nFold {fold_idx + 1}:")

        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        train_data = all_data[train_indices]
        train_labels_fold = all_labels[train_indices]
        train_names = [sample_names[i] for i in train_indices]

        val_data = all_data[val_indices]
        val_labels_fold = all_labels[val_indices]
        val_names = [sample_names[i] for i in val_indices]

        train_dist = pd.Series(train_labels_fold).value_counts().sort_index()
        val_dist = pd.Series(val_labels_fold).value_counts().sort_index()

        print(f"  Train set: {len(train_labels_fold)} samples")
        print(f"    Label distribution: {dict(train_dist)}")
        print(f"  Validation set: {len(val_labels_fold)} samples")
        print(f"    Label distribution: {dict(val_dist)}")

        np.save(os.path.join(fold_dir, 'train_data.npy'), train_data)
        with open(os.path.join(fold_dir, 'train_label.pkl'), 'wb') as f:
            pickle.dump((train_labels_fold.tolist(), train_names), f)

        np.save(os.path.join(fold_dir, 'val_data.npy'), val_data)
        with open(os.path.join(fold_dir, 'val_label.pkl'), 'wb') as f:
            pickle.dump((val_labels_fold.tolist(), val_names), f)

        print(f"  Saved to: {fold_dir}")

    print("\n✓ 5-fold cross-validation dataset generated successfully!")
    print(f"Output directory: {output_dir}")

    print("\nGenerating configuration files...")
    config_dir = '/home/weiping/st-gcn-master/st-gcn-master/config/last'
    os.makedirs(config_dir, exist_ok=True)

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
        print(f"Created config file: {config_file}")

    print("\n✓ Configuration files generated successfully!")
    print(f"Config directory: {config_dir}")
    print("\nYou can start training with the following command:")
    print("python3 main.py recognition -c config/last/train_fold1.yaml")


if __name__ == "__main__":
    create_last_dataset()
