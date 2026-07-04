import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from google.colab import drive

#Mount Google Drive
drive.mount('/content/drive')

labels_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Resized_Frame_Labels'
preds_base_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Pose_Predictions'

enhancement_methods = [
    "Original_Resized_Frames",
    "CLAHE_Frames",
    "CNN_Frames",
    "Retinex_Frames",
    "Transformer_Frames",
    "UNET_Frames",
    "WhiteBalance_Frames"
]

#Limbs for limb length consistency 
LIMBS = [
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle')
]

EVAL_JOINTS = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]

CONF_THRESHOLD = 0.3
REF_SCALE = 256.0  # Image size is 256x256
#There are several frames with zero and only a few visible joints, using threshold of 8+ visibile joints
MIN_VISIBLE_GT_JOINTS = 8  # Only evaluate accuracy on frames with >= 8 visible GT joints

#Retrieving joint annotations (ground truth + predictions from each enhancement method)
def load_keypoints(filepath):
    kpts = {}
    if not os.path.exists(filepath): return kpts
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if lines and 'joint' in lines[0].lower():
        lines = lines[1:]
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            j_name = parts[0]
            x, y = float(parts[1]), float(parts[2])
            val = float(parts[3])
            kpts[j_name] = {'x': x, 'y': y, 'val': val}
    return kpts

def is_visible_gt(joint_data):
    x = joint_data['x']
    y = joint_data['y']
    v = joint_data['val']
    return v > 0 and not (x == 0 and y == 0)

results = []

label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
if not label_files:
    print("No label files found in:", labels_dir)

for method in enhancement_methods:
    preds_dir = os.path.join(preds_base_dir, method)
    #Metrics
    errors = []
    confidences = []
    confident_kpts_count = 0
    total_kpts_count = 0
    pose_failures = 0
    num_supervised_frames = 0
    num_visible_eval_joints = 0
    #Temporal motion and limb length
    prev_kpts = None
    prev_video_id = None
    motions = []
    video_limb_lengths = defaultdict(lambda: defaultdict(list))

    for gt_file in label_files:
        filename = os.path.basename(gt_file)
        video_id = filename.split('_frame_')[0] if '_frame_' in filename else 'unknown_video'
        if video_id != prev_video_id:
            prev_kpts = None
            prev_video_id = video_id

        pred_file = os.path.join(preds_dir, filename)
        pred_kpts = load_keypoints(pred_file)
        gt_kpts = load_keypoints(gt_file)
        #If keypoint is not predicted it counts as a failure
        if not pred_kpts:
            pose_failures += 1
            prev_kpts = None
            continue
        frame_confident_count = 0
        visible_gt_joints = [
            j for j in EVAL_JOINTS
            if j in gt_kpts and is_visible_gt(gt_kpts[j])
        ]
        valid_for_supervised_eval = len(visible_gt_joints) >= MIN_VISIBLE_GT_JOINTS

        frame_errors = []

        for j_name in EVAL_JOINTS:
            if j_name not in pred_kpts:
                continue
            p_data = pred_kpts[j_name]
            score = p_data['val']
            confidences.append(score)
            total_kpts_count += 1
            if score >= CONF_THRESHOLD:
                frame_confident_count += 1
                confident_kpts_count += 1

            #Ground Truth metrics (only if GT exists and is visible)
            if j_name in gt_kpts and is_visible_gt(gt_kpts[j_name]):
                gx, gy = gt_kpts[j_name]['x'], gt_kpts[j_name]['y']
                px, py = p_data['x'], p_data['y']
                dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                frame_errors.append(dist)

            #Frame-to-Frame Motion 
            if prev_kpts and j_name in prev_kpts:
                if score >= CONF_THRESHOLD and prev_kpts[j_name]['val'] >= CONF_THRESHOLD:
                    dist = np.sqrt((p_data['x'] - prev_kpts[j_name]['x'])**2 +
                                   (p_data['y'] - prev_kpts[j_name]['y'])**2)
                    motions.append(dist)

        #Add errors only if there are enough visible GT joints
        if valid_for_supervised_eval:
            errors.extend(frame_errors)
            num_supervised_frames += 1
            num_visible_eval_joints += len(frame_errors)

        #Pose failure if less than 5 confident keypoints in a frame
        if frame_confident_count < 5:
            pose_failures += 1

        #Limb length calculation
        for j1, j2 in LIMBS:
            if j1 in pred_kpts and j2 in pred_kpts:
                if pred_kpts[j1]['val'] >= CONF_THRESHOLD and pred_kpts[j2]['val'] >= CONF_THRESHOLD:
                    ll = np.sqrt((pred_kpts[j1]['x'] - pred_kpts[j2]['x'])**2 +
                                 (pred_kpts[j1]['y'] - pred_kpts[j2]['y'])**2)
                    video_limb_lengths[video_id][f"{j1}-{j2}"].append(ll)

        prev_kpts = pred_kpts

    #Aggregate Metrics
    errors = np.array(errors)
    mean_pixel_error = np.mean(errors) if len(errors) > 0 else np.nan
    nme = mean_pixel_error / REF_SCALE if not np.isnan(mean_pixel_error) else np.nan
    pck_05 = np.mean(errors <= (0.05 * REF_SCALE)) * 100 if len(errors) > 0 else np.nan
    pck_10 = np.mean(errors <= (0.10 * REF_SCALE)) * 100 if len(errors) > 0 else np.nan
    pck_20 = np.mean(errors <= (0.20 * REF_SCALE)) * 100 if len(errors) > 0 else np.nan
    mean_conf = np.mean(confidences) if confidences else 0
    pct_confident = (confident_kpts_count / total_kpts_count) * 100 if total_kpts_count > 0 else 0

    #Calculate failure rate against the master list of label files
    num_frames = len(label_files)
    failure_rate = (pose_failures / num_frames) * 100 if num_frames > 0 else 0
    mean_motion = np.mean(motions) if motions else np.nan

    #Average limb length coefficient of variation (std / mean)
    video_cvs = []
    for vid_id, limbs_dict in video_limb_lengths.items():
        for limb, lengths in limbs_dict.items():
            if len(lengths) > 1:
                video_cvs.append(np.std(lengths) / np.mean(lengths))
    mean_limb_cv = np.mean(video_cvs) if video_cvs else np.nan

    results.append({
        "Method": method,
        "Supervised Frames": num_supervised_frames,
        "Visible Eval Joints": num_visible_eval_joints,
        "PCK@0.05 (%)": pck_05,
        "PCK@0.10 (%)": pck_10,
        "PCK@0.20 (%)": pck_20,
        "Mean Pixel Err": mean_pixel_error,
        "NME (%)": nme * 100 if not np.isnan(nme) else np.nan,
        "Mean Conf": mean_conf,
        "% Kpts >= 0.3": pct_confident,
        "Failure Rate (%)": failure_rate,
        "Frame-to-Frame Motion (px)": mean_motion,
        "Limb CV (Stability)": mean_limb_cv
    })

df_results = pd.DataFrame(results)
df_results = df_results.round(3)
display(df_results)

