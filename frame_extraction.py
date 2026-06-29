from google.colab import drive
import cv2
import os
import glob

drive.mount('/content/drive')
input_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Raw_Videos'
output_dir = '/content/drive/MyDrive/SwimXYZ/PilotStudy/Video_Frames'

#Creating output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

#Find all .webm files in the input directory (Should be ten for pilot study)
video_files = glob.glob(os.path.join(input_dir, '*.webm'))
print(f"Found {len(video_files)} video files.")

#Capturing every 5th frame in each of the videos, each video should have 300 frames so 60 frames are extracted
for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            output_filename = f"{video_name}_frame_{frame_count:05d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Saved {saved_count} frames")

print("\nFrame extraction complete")
