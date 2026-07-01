from google.colab import drive
import os
import glob
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img


'''
This files main purpose is to load in the pre-trained deep learning UIE methods 
and run them through the 600 chosen SwimXYZ frames. 
Input and Output are folders within Google Drive
'''


#Mount Google Drive
drive.mount('/content/drive')

#Custom residual output for UNET and Transformer
@tf.keras.utils.register_keras_serializable(name='residual_output')
def residual_output(inputs):
    return inputs[0] + inputs[1]

#Google drive folders
base_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy'
model_path = os.path.join(base_dir, 'UIE_Methods', 'lightweight_Transformer_Small_best.keras')
input_dir = os.path.join(base_dir, 'Video_Frames', 'Raw_Frames')
output_dir = os.path.join(base_dir, 'Video_Frames', 'CLAHE_Frames')

#Creating output directory (only if it doesn't exist)
os.makedirs(output_dir, exist_ok=True)

#Load the model
#(compile=False skips looking for the custom training functions like uie_loss)
print(f"Loading model from {model_path}...")
#Pass the custom function in the custom_objects dictionary (Only for the UNET & Transformer)
#model = load_model(model_path, compile=False) #Only for Lightweight CNN
model = load_model(model_path, compile=False, custom_objects={'residual_output': residual_output}) #Only for UNET & Transformer
print("Model loaded successfully.")

#Retrieve all images that will undergo UIE
image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
print(f"Found {len(image_paths)} images to process in {input_dir}.")

#Process each image
for img_path in image_paths:
    filename = os.path.basename(img_path)
    orig_img = cv2.imread(img_path)
    #OpenCV loads images in BGR format, so convert to RGB
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    #Resize image to 256x256
    resized_img = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_AREA)
    #Normalize image
    img_array = resized_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    #Generate enhanced image
    enhanced_img_array = model.predict(img_array, verbose=0)
    #Remove batch dimension
    enhanced_img_array = np.squeeze(enhanced_img_array, axis=0)
    #Save image (save_img automatically handles scaling back to 0-255 if needed)
    save_img(os.path.join(output_dir, filename), enhanced_img_array)

print(f"Processing complete. Enhanced images saved to {output_dir}")
