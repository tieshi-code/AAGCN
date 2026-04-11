#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def normalize_dp_name(dp_raw: str) -> str:
    if dp_raw is None:
        return None
    dp = str(dp_raw).strip()
    dp = dp.replace("Data point ", "Data point")
    if "." in dp:
        dp = dp.split(".")[0]
    return dp


def load_excel_labels(excel_file):
    print("Loading Excel label file...")
    df = pd.read_excel(excel_file)

    label_map = {}
    amputee_map = {}
    
    for _, row in df.iterrows():
        lk = str(row['LK']).strip()
        dp = normalize_dp_name(str(row['Data_Point']))
        key = (lk, dp)
        label_map[key] = int(row['整合标签'])
        
        amputee_label = str(row.get('左右截肢腿', '')).strip()
        if amputee_label == '右腿截肢':
            amputee_map[key] = 'right_leg'
        elif amputee_label == '左腿截肢':
            amputee_map[key] = 'left_leg'
        elif amputee_label == '左腿和右腿都截肢':
            amputee_map[key] = 'both_legs'
        else:
            amputee_map[key] = 'none'

    print(f"Loaded {len(label_map)} label mappings")
    print(f"Loaded {len(amputee_map)} amputee label mappings")
    return label_map, amputee_map


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
        print(f"Failed to process video data {json_file}: {e}")
        return None, None


def create_last_dataset_with_amputee():
    print("=== Creating last dataset (with amputee labels) ===\n")

    excel_file = '/home/weiping/st-gcn-master/st-gcn-master/five_tests_integrated_labels_detailed.xlsx'
    alphapose_dirs = [
        Path('/home/weiping/AlphaPose-master/video_data/single_straight_walk/output'),
        Path('/home/weiping/AlphaPose-master/video_data/single_tug/output')
    ]
    output_dir = '/home/weiping/st-gcn-master/st-gcn-master/data/last_amputee'

    label_map, amputee_map = load_excel_labels(excel_file)

    video_files = find_video_files(alphapose_dirs)

    print("\nProcessing video data...")
    data_list = []
    label_list = []
    amputee_list = []
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
                print(f"Processed {processed_count}/{len(video_files)} videos")

    print(f"\nSuccessfully processed {len(data_list)} video files")

    if not data_list:
        print("Failed to process any video files")
        return

    all_data = np.stack(data_list, axis=0)
    all_labels = np.array(label_list)

    print(f"Data shape: {all_data.shape}")
    print(f"Label distribution: {dict(sorted(Counter(all_labels).items()))}")
    print(f"Amputee label distribution: {dict(Counter(amputee_list))}")

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'full_data.npy'), all_data)
    with open(os.path.join(output_dir, 'full_labels.pkl'), 'wb') as f:
        pickle.dump((all_labels.tolist(), sample_names), f)
    
    with open(os.path.join(output_dir, 'full_amputee_labels.pkl'), 'wb') as f:
        pickle.dump((amputee_list, sample_names), f)

    print(f"Full dataset saved to: {output_dir}")

    print("\nGenerating 5-fold cross-validation...")
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
        print(f"Fold {fold_idx + 1}: Train {len(train_labels)}, Val {len(val_labels)}")
        print(f"  Train distribution: {dict(sorted(train_dist.items()))}")
        print(f"  Val distribution: {dict(sorted(val_dist.items()))}")

    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    create_last_dataset_with_amputee()
