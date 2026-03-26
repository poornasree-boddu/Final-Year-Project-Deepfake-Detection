import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
model.eval()
model.to(device)

final_conv = model.inception5b

gradients = []
activations = []


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


def forward_hook(module, module_input, output):
    activations.append(output)


final_conv.register_forward_hook(forward_hook)
final_conv.register_full_backward_hook(backward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


frame_dir = "frames/fake"
fake_frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")])
if not fake_frames:
    raise FileNotFoundError("No .jpg frames found in frames/fake")

image_path = os.path.join(frame_dir, fake_frames[0])
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

output = model(input_tensor)
pred_class = output.argmax(dim=1)

model.zero_grad()
output[0, pred_class.item()].backward()

gradient = gradients[-1].detach().cpu().numpy()[0]
activation = activations[-1].detach().cpu().numpy()[0]

weights = np.mean(gradient, axis=(1, 2))
cam = np.zeros(activation.shape[1:], dtype=np.float32)

for i, weight in enumerate(weights):
    cam += weight * activation[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam - np.min(cam)
cam = cam / (np.max(cam) + 1e-8)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
original = cv2.imread(image_path)
original = cv2.resize(original, (224, 224))
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/gradcam_result.jpg"
cv2.imwrite(out_path, overlay)

print(f"Input frame: {image_path}")
print(f"Predicted class index: {pred_class.item()}")
print(f"Grad-CAM visualization saved to {out_path}")
