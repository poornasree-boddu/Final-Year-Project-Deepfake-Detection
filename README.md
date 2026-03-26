# 🧠 Multimodal Deepfake Detection System

A unified deep learning system to detect deepfake content by analyzing **images, videos, and audio** simultaneously. This project leverages multiple AI models along with a fusion mechanism to provide highly accurate and robust deepfake detection.

---

## 🚀 Features

* 🔍 Detects deepfakes in:

  * Images (JPG, PNG)
  * Videos (MP4)
  * Audio (WAV)
* 🧠 Deep learning models for each modality
* 🔗 Fusion model combines all predictions
* 🔥 Grad-CAM visualization for explainability (images)
* 🌐 Interactive web app using Streamlit
* 📊 High accuracy with multimodal approach

---

## 🏗️ System Architecture

The system consists of four main modules:

### 1. Image Module

* Model: EfficientNet-B0
* Detects visual deepfake artifacts
* Provides Grad-CAM heatmaps

### 2. Video Module

* Model: GoogLeNet + BiLSTM
* Captures spatial and temporal inconsistencies

### 3. Audio Module

* Model: CNN on Mel Spectrograms
* Detects synthetic speech

### 4. Fusion Module

* Combines outputs from all modalities
* Produces final prediction

---

## 📊 Results

| Module   | Accuracy  |
| -------- | --------- |
| Image    | 93%       |
| Audio    | 91%       |
| Video    | 89%       |
| ⭐ Fusion | **94.5%** |

---

## 🛠️ Tech Stack

* Python
* PyTorch
* OpenCV
* Librosa
* Streamlit
* NumPy

---

## 📁 Project Structure

```id="m2f3b1"
project/
│
├── datasets/
├── image_module/
├── audio_module/
├── video_module/
├── fusion_module/
│
├── models/
│   ├── image_model.pth
│   ├── audio_model.pth
│   └── video_model.pth
│
├── demo_media/
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

* Create and activate a Python virtual environment
* Install dependencies:

```bash id="6a7k9s"
pip install -r requirements.txt
```

* Add your datasets and trained model files locally:

Place datasets under:

```id="2bx7mk"
datasets/
```

Place model weights under:

```id="9pl2xr"
models/
```

Example:

```id="z8t4vq"
models/
├── image_model.pth
├── audio_model.pth
└── video_model.pth
```

---

## ▶️ Run The App

```bash id="y6c3jh"
streamlit run app.py
```

---

## 📸 How It Works

1. Upload image, video, or audio
2. Model processes input
3. Each module gives prediction
4. Fusion combines results
5. Final output with confidence is shown

---

## 🧪 Supported Formats

| Type  | Formats  |
| ----- | -------- |
| Image | JPG, PNG |
| Video | MP4      |
| Audio | WAV      |

---

## 📌 Key Highlights

* Multimodal detection (image + video + audio)
* Higher accuracy than single models
* Explainable AI with Grad-CAM
* Modular design
* Easy-to-use interface

---

## 🔮 Future Enhancements

* Transformer-based models
* Real-time detection
* Attention-based fusion
* Text deepfake detection
* Adversarial robustness

---

## 👨‍💻 Authors

* B V Poorna Sree
* D Yashwanth
* G Raja
* A Hemanth Kumar
* Under the guidence of Dr.Ravi Kiran (Prof. of Raghu Institute of Technology)
---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgement

Developed as part of B.Tech Final Year Project in Computer Science and Engineering (Data Science).

---
