import os
import cv2
from tqdm import tqdm

# Paths
REAL_VIDEO_PATH = "dataset/real_subset"
FAKE_VIDEO_PATH = "dataset/fake_subset"

REAL_FRAME_PATH = "frames/real"
FAKE_FRAME_PATH = "frames/fake"

FRAMES_PER_VIDEO = 40


def extract_frames(video_path, output_folder, label):
    video_name = os.path.basename(video_path).split(".")[0]
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // FRAMES_PER_VIDEO)

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0 and saved < FRAMES_PER_VIDEO:
            frame_filename = f"{label}_{video_name}_{saved}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()


def process_videos(video_folder, output_folder, label):
    video_list = os.listdir(video_folder)
    for video_file in tqdm(video_list):
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, output_folder, label)


if __name__ == "__main__":
    print("Extracting REAL videos...")
    process_videos(REAL_VIDEO_PATH, REAL_FRAME_PATH, "real")

    print("Extracting FAKE videos...")
    process_videos(FAKE_VIDEO_PATH, FAKE_FRAME_PATH, "fake")

    print("Frame extraction completed.")