import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

FRAME_PATH = "frames/real_faces"
FEATURE_PATH = "features"

FRAMES_PER_VIDEO = 20


def extract_features_from_folder(input_label, output_label):
    frame_folder = os.path.join(FRAME_PATH if input_label == "real" else "frames/fake_faces")
    save_folder = os.path.join(FEATURE_PATH, output_label)
    os.makedirs(save_folder, exist_ok=True)

    video_dict = {}

    for frame_file in os.listdir(frame_folder):
        parts = frame_file.split("_")
        video_name = "_".join(parts[1:-1])
        video_dict.setdefault(video_name, []).append(frame_file)

    for video_name, frame_files in tqdm(video_dict.items()):
        frame_files = sorted(
            frame_files,
            key=lambda name: int(os.path.splitext(name)[0].split("_")[-1]),
        )
        features = []

        for frame_file in frame_files:
            img_path = os.path.join(frame_folder, frame_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    feature = model(image)
                    features.append(feature.cpu().numpy())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if len(features) > 0:
            features = np.vstack(features)
            np.save(os.path.join(save_folder, f"{video_name}.npy"), features)


if __name__ == "__main__":
    print("Extracting REAL face features...")
    extract_features_from_folder("real", "real")

    print("Extracting FAKE face features...")
    extract_features_from_folder("fake", "fake")

    print("Feature extraction completed.")
