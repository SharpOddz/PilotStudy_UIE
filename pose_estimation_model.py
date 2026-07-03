#Download RTMPose-M Model
!pip uninstall -y mmcv mmengine mmpose mmdet openmim
!pip install -q -U pip setuptools wheel
!pip install -q rtmlib opencv-python opencv-contrib-python
!pip uninstall -y onnxruntime onnxruntime-gpu
!pip install -q onnxruntime

#Load in the Model
import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose, draw_skeleton
from google.colab import drive

providers = ort.get_available_providers()
device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"

print("Using device:", device)

RTMPOSE_M_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)

pose_model = RTMPose(
    onnx_model=RTMPOSE_M_URL,
    model_input_size=(192, 256),  # RTMPose uses width, height
    backend="onnxruntime",
    device=device
)

#Mount Google Drive
drive.mount('/content/drive')

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

#Google Drive Directories
base_images_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Video_Frames'
base_preds_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Pose_Predictions'

enhancement_methods = [
    "WhiteBalance_Frames",
    "UNET_Frames",
    "Transformer_Frames",
    "Retinex_Frames",
    "Original_Resized_Frames",
    "CNN_Frames",
    "CLAHE_Frames"
]

COCO_KEYPOINTS = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]

SWIMXYZ_OUTPUT_ORDER = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

#For each enhancement method, load in the enhanced images, sort them,
#and then run inference
for method in enhancement_methods:
    print(f"\nProcessing {method}...")
    images_dir = os.path.join(base_images_dir, method)
    preds_dir = os.path.join(base_preds_dir, method)
    os.makedirs(preds_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, '*.jpg')) +
        glob.glob(os.path.join(images_dir, '*.jpeg')) +
        glob.glob(os.path.join(images_dir, '*.png')) +
        glob.glob(os.path.join(images_dir, '*.JPG')) +
        glob.glob(os.path.join(images_dir, '*.PNG'))
    )
    print(f"Found {len(image_paths)} images to process in {method}.")

    if len(image_paths) == 0:
        continue

    #Go through all images and run inference
    for img_path in tqdm(image_paths, desc=f"Running RTMPose on {method}"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        #Full-frame bbox: [x1, y1, x2, y2]
        bboxes = [[0, 0, w, h]]

        keypoints, scores = pose_model(img, bboxes=bboxes)
        if len(keypoints) == 0:
            continue

        kpts = keypoints[0]
        scrs = scores[0]
        #Neck and mid hip have to be manually calculated
        neck = 0.5 * (kpts[COCO_KEYPOINTS.index("LShoulder")] +
                      kpts[COCO_KEYPOINTS.index("RShoulder")])
        neck_score = min(scrs[COCO_KEYPOINTS.index("LShoulder")],
                         scrs[COCO_KEYPOINTS.index("RShoulder")])

        midhip = 0.5 * (kpts[COCO_KEYPOINTS.index("LHip")] +
                        kpts[COCO_KEYPOINTS.index("RHip")])
        midhip_score = min(scrs[COCO_KEYPOINTS.index("LHip")],
                           scrs[COCO_KEYPOINTS.index("RHip")])

        #Creating the format that best alligns with the SwimXYZ format
        swimxyz_kpts = {}
        for i, name in enumerate(COCO_KEYPOINTS):
            swimxyz_kpts[name] = (kpts[i][0], kpts[i][1], scrs[i])

        swimxyz_kpts["Neck"] = (neck[0], neck[1], neck_score)
        swimxyz_kpts["MidHip"] = (midhip[0], midhip[1], midhip_score)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        pred_path = os.path.join(preds_dir, f"{base_name}.txt")

        with open(pred_path, 'w') as f:
            f.write("joint,x,y,score\n")
            for joint in SWIMXYZ_OUTPUT_ORDER:
                if joint in swimxyz_kpts:
                    x, y, s = swimxyz_kpts[joint]
                    f.write(f"{joint},{x:.4f},{y:.4f},{s:.4f}\n")

print(f"\nAll predictions saved successfully to subfolders in: {base_preds_dir}")
