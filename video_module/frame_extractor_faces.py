import os
import cv2
from tqdm import tqdm

REAL_VIDEO_PATH = "dataset/real_subset"
FAKE_VIDEO_PATH = "dataset/fake_subset"

REAL_FRAME_PATH = "frames/real_faces"
FAKE_FRAME_PATH = "frames/fake_faces"

FRAMES_PER_VIDEO = 20

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face_region(frame, face_cascade, padding=0.2):
    """Extract largest face region with padding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        return frame[y1:y2, x1:x2]
    return None


def extract_frames(video_path, output_folder, label, face_cascade):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
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
            face_region = extract_face_region(frame, face_cascade)
            if face_region is not None:
                frame_filename = f"{label}_{video_name}_{saved}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, face_region)
                saved += 1

        count += 1

    cap.release()


def process_videos(video_folder, output_folder, label, face_cascade):
    os.makedirs(output_folder, exist_ok=True)
    video_list = os.listdir(video_folder)
    for video_file in tqdm(video_list):
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, output_folder, label, face_cascade)


if __name__ == "__main__":
    os.makedirs(REAL_FRAME_PATH, exist_ok=True)
    os.makedirs(FAKE_FRAME_PATH, exist_ok=True)

    print("Extracting REAL video frames (face-cropped)...")
    process_videos(REAL_VIDEO_PATH, REAL_FRAME_PATH, "real", face_cascade)

    print("Extracting FAKE video frames (face-cropped)...")
    process_videos(FAKE_VIDEO_PATH, FAKE_FRAME_PATH, "fake", face_cascade)

    print("Frame extraction completed.")
