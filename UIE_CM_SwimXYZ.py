import os
import glob
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import save_img

'''
This files main purpose is to perform classical UIE
on the 600 chosen SwimXYZ frames. 
Input and Output are folders within Google Drive
'''


#Gray World White Balance
def gray_world_white_balance(img):
    img_float = img.astype(np.float32)
    avg_r = np.mean(img_float[:, :, 2])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 0])
    avg_gray = (avg_r + avg_g + avg_b) / 3
    img_float[:, :, 2] *= (avg_gray / avg_r)
    img_float[:, :, 1] *= (avg_gray / avg_g)
    img_float[:, :, 0] *= (avg_gray / avg_b)
    return np.clip(img_float, 0, 255).astype(np.uint8)

#Retinex
def simple_retinex(img, sigma=50):
    img_float = np.float32(img) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(blurred)
    #Normalize each channel to 0-255
    for i in range(img.shape[2]):
        min_val = np.min(retinex[:, :, i])
        max_val = np.max(retinex[:, :, i])
        retinex[:, :, i] = (retinex[:, :, i] - min_val) / (max_val - min_val) * 255.0
    return np.clip(retinex, 0, 255).astype(np.uint8)

#Google drive folders
base_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy'
input_dir = os.path.join(base_dir, 'Video_Frames', 'Raw_Frames')
output_dir = os.path.join(base_dir, 'Video_Frames', 'Retinex_Frames')
#Creating output directory
os.makedirs(output_dir, exist_ok=True)
#Retrieve all images
image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        continue
    #Resize image to 256x256
    resized_bgr = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)
    #Apply Gray World White Balance
    #balanced_bgr = gray_world_white_balance(resized_bgr)
    #Apply Retinex
    balanced_bgr = simple_retinex(resized_bgr)
    #Convert BGR to RGB for Keras save_img
    final_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    #Save image
    save_img(os.path.join(output_dir, filename), final_rgb)
print(f"Processing complete. Enhanced images saved to {output_dir}")
