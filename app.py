"""Streamlit UI for multimodal deepfake detection."""

from __future__ import annotations

import os
import tempfile
from glob import glob
from typing import List, Tuple

import streamlit as st

from fusion_module.inference import predict_multimodal


IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
VIDEO_EXTENSIONS = ["mp4"]
AUDIO_EXTENSIONS = ["wav"]


def _save_uploaded_file(uploaded_file) -> str:
    temp_dir = tempfile.mkdtemp(prefix="deepfake_ui_")
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as handle:
        handle.write(uploaded_file.read())
    return file_path


def _list_files(patterns: List[str], max_items: int = 20) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(pattern))
    files = sorted(set(files))
    return files[:max_items]


def _demo_image_files() -> List[str]:
    files = _list_files(["demo_media/images/*.jpg", "demo_media/images/*.jpeg", "demo_media/images/*.png"])
    if files:
        return files
    return _list_files(
        [
            "datasets/image_dataset/Test/Fake/*.jpg",
            "datasets/image_dataset/Test/Real/*.jpg",
        ]
    )


def _demo_video_files() -> List[str]:
    files = _list_files(["demo_media/videos/fake/*.mp4", "demo_media/videos/real/*.mp4"])
    if files:
        return files
    return _list_files(["datasets/video_dataset/fake_subset/*.mp4", "datasets/video_dataset/real_subset/*.mp4"])


def _demo_audio_files() -> List[str]:
    demo_files = _list_files(["demo_media/audio/*.wav"])
    dataset_files = _list_files(
        [
            "datasets/audio_dataset/testing/fake/*.wav",
            "datasets/audio_dataset/testing/real/*.wav",
        ],
        max_items=200,
    )
    all_files = sorted(set(demo_files + dataset_files))
    return all_files


def _split_fake_real(files: List[str]) -> Tuple[List[str], List[str]]:
    fake_files: List[str] = []
    real_files: List[str] = []
    unknown_files: List[str] = []

    for path in files:
        normalized = path.replace("\\", "/").lower()
        basename = os.path.basename(path).lower()
        if "/fake/" in normalized or "fake" in basename:
            fake_files.append(path)
        elif "/real/" in normalized or "real" in basename:
            real_files.append(path)
        else:
            unknown_files.append(path)

    # If files are unlabeled (common for audio demo folder), split them evenly.
    for idx, path in enumerate(unknown_files):
        if idx % 2 == 0:
            fake_files.append(path)
        else:
            real_files.append(path)

    return fake_files, real_files


def _reset_detection_state() -> None:
    st.session_state.detection_result = None
    st.session_state.selected_image = None
    st.session_state.selected_video = None
    st.session_state.selected_audio = None
    for key in ["img_upload", "vid_upload", "aud_upload"]:
        if key in st.session_state:
            del st.session_state[key]


def _confidence_reliability(confidence_percent: float) -> str:
    if confidence_percent >= 90:
        return "High"
    if confidence_percent >= 75:
        return "Medium"
    return "Low"


def main() -> None:
    st.set_page_config(page_title="Multimodal Deepfake Detection", layout="wide")

    st.title("🧠 Multimodal Deepfake Detection System")
    st.write("Detect deepfake media with image, video, and audio models plus multimodal fusion.")

    demo_images = _demo_image_files()
    demo_videos = _demo_video_files()
    demo_audios = _demo_audio_files()

    audio_fake_pool, audio_real_pool = _split_fake_real(demo_audios)
    if len(audio_fake_pool) < 5:
        audio_fake_pool.extend(_list_files(["datasets/audio_dataset/testing/fake/*.wav"], max_items=50))
    if len(audio_real_pool) < 5:
        audio_real_pool.extend(_list_files(["datasets/audio_dataset/testing/real/*.wav"], max_items=50))
    audio_fake_pool = sorted(set(audio_fake_pool))
    audio_real_pool = sorted(set(audio_real_pool))

    # Initialize session state for tracking active modality and results
    if "active_modality" not in st.session_state:
        st.session_state.active_modality = "image"
    if "detection_result" not in st.session_state:
        st.session_state.detection_result = None
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = None
    if "selected_audio" not in st.session_state:
        st.session_state.selected_audio = None

    selected_modality = st.radio(
        "Modality",
        options=["image", "video", "audio"],
        format_func=lambda x: {"image": "🖼 Image", "video": "🎥 Video", "audio": "🎙 Audio"}[x],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_modality != st.session_state.active_modality:
        st.session_state.active_modality = selected_modality
        _reset_detection_state()

    # IMAGE PANEL
    if selected_modality == "image":
        
        st.write("**Select an image source**")
        image_mode = st.radio("Choose:", options=["📥 Download & Upload", "🎬 Use Demo"], horizontal=True, key="img_mode", label_visibility="collapsed")
        
        if image_mode == "📥 Download & Upload":
            st.markdown("**📂 Download sample images to test:**")
            fake_imgs, real_imgs = _split_fake_real(demo_images)
            fake_imgs = fake_imgs[:5]
            real_imgs = real_imgs[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_imgs):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_img_fake_{idx}"
                        )
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_imgs):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_img_real_{idx}"
                        )
            
            st.divider()
            st.markdown("**📤 Upload your image:**")
            uploaded_image = st.file_uploader("Upload image", type=IMAGE_EXTENSIONS, key="img_upload", label_visibility="collapsed")
            if uploaded_image:
                st.session_state.selected_image = _save_uploaded_file(uploaded_image)
        
        else:  # Demo mode
            st.markdown("**🎬 Demo samples - Click USE to select:**")
            fake_imgs, real_imgs = _split_fake_real(demo_images)
            fake_imgs = fake_imgs[:5]
            real_imgs = real_imgs[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_imgs):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_img_fake_{idx}", use_container_width=True):
                        st.session_state.selected_image = fpath
                        st.rerun()
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_imgs):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_img_real_{idx}", use_container_width=True):
                        st.session_state.selected_image = fpath
                        st.rerun()
        
        if st.session_state.selected_image:
            st.image(st.session_state.selected_image, caption="Selected Image", use_container_width=True)

    # VIDEO PANEL
    elif selected_modality == "video":
        
        st.write("**Select a video source**")
        video_mode = st.radio("Choose:", options=["📥 Download & Upload", "🎬 Use Demo"], horizontal=True, key="vid_mode", label_visibility="collapsed")
        
        if video_mode == "📥 Download & Upload":
            st.markdown("**📂 Download sample videos to test:**")
            fake_vids, real_vids = _split_fake_real(demo_videos)
            fake_vids = fake_vids[:5]
            real_vids = real_vids[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_vids):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_vid_fake_{idx}"
                        )
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_vids):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_vid_real_{idx}"
                        )
            
            st.divider()
            st.markdown("**📤 Upload your video:**")
            uploaded_video = st.file_uploader("Upload video", type=VIDEO_EXTENSIONS, key="vid_upload", label_visibility="collapsed")
            if uploaded_video:
                st.session_state.selected_video = _save_uploaded_file(uploaded_video)
        
        else:  # Demo mode
            st.markdown("**🎬 Demo samples - Click USE to select:**")
            fake_vids, real_vids = _split_fake_real(demo_videos)
            fake_vids = fake_vids[:5]
            real_vids = real_vids[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_vids):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_vid_fake_{idx}", use_container_width=True):
                        st.session_state.selected_video = fpath
                        st.rerun()
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_vids):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_vid_real_{idx}", use_container_width=True):
                        st.session_state.selected_video = fpath
                        st.rerun()
        
        if st.session_state.selected_video:
            st.video(st.session_state.selected_video)

    # AUDIO PANEL
    else:
        
        st.write("**Select an audio source**")
        audio_mode = st.radio("Choose:", options=["📥 Download & Upload", "🎬 Use Demo"], horizontal=True, key="aud_mode", label_visibility="collapsed")
        
        if audio_mode == "📥 Download & Upload":
            st.markdown("**📂 Download sample audio to test:**")
            fake_auds = audio_fake_pool[:5]
            real_auds = audio_real_pool[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_auds):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_aud_fake_{idx}"
                        )
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_auds):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {os.path.basename(fpath)[:20]}",
                            data=f.read(),
                            file_name=os.path.basename(fpath),
                            key=f"dl_aud_real_{idx}"
                        )
            
            st.divider()
            st.markdown("**📤 Upload your audio:**")
            uploaded_audio = st.file_uploader("Upload audio", type=AUDIO_EXTENSIONS, key="aud_upload", label_visibility="collapsed")
            if uploaded_audio:
                st.session_state.selected_audio = _save_uploaded_file(uploaded_audio)
        
        else:  # Demo mode
            st.markdown("**🎬 Demo samples - Click USE to select:**")
            fake_auds = audio_fake_pool[:5]
            real_auds = audio_real_pool[:5]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Fake Samples:**")
                for idx, fpath in enumerate(fake_auds):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_aud_fake_{idx}", use_container_width=True):
                        st.session_state.selected_audio = fpath
                        st.rerun()
            
            with col2:
                st.markdown("**🟢 Real Samples:**")
                for idx, fpath in enumerate(real_auds):
                    if st.button(f"✓ Use - {os.path.basename(fpath)[:20]}", key=f"use_aud_real_{idx}", use_container_width=True):
                        st.session_state.selected_audio = fpath
                        st.rerun()
        
        if st.session_state.selected_audio:
            st.audio(st.session_state.selected_audio)

    st.divider()

    run_disabled = not any([st.session_state.selected_image, st.session_state.selected_video, st.session_state.selected_audio])
    if st.button("▶️ Run Detection", type="primary", disabled=run_disabled, use_container_width=True):
        progress = st.progress(0, text="Analyzing...")
        try:
            with st.spinner("Running analysis..."):
                progress.progress(50)

                result = predict_multimodal(
                    video_path=st.session_state.selected_video,
                    image_path=st.session_state.selected_image,
                    audio_path=st.session_state.selected_audio,
                    weights={"video": 1.0, "image": 1.0, "audio": 1.0},
                    threshold=0.5,
                    generate_video_gradcam=True,
                )

                st.session_state.detection_result = result
                progress.progress(100)

            st.success("✅ Analysis Complete")

        except Exception as exc:
            st.error(f"❌ Detection failed: {str(exc)}")

    # Display results only if we have them
    if st.session_state.detection_result:
        result = st.session_state.detection_result
        
        st.subheader("📊 Result")
        
        pred = result.get("prediction", "Unknown")
        confidence = float(result.get("confidence", 0.0))
        fake_probability = float(result.get("probabilities", {}).get("fake", 0.0))
        reliability = _confidence_reliability(confidence)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", pred, delta=f"{fake_probability:.1f}% Fake" if pred == "Fake" else "Real")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Reliability", reliability)
        col4.metric("Fake Score", f"{fake_probability:.1f}%")
        
        st.progress(int(max(0.0, min(100.0, fake_probability))) / 100.0)
        
        # Show details in expander
        with st.expander("📈 Detailed Analysis"):
            modal_results = result.get("modal_results", {})
            
            st.markdown("**Per-Modality Breakdown:**")
            
            if modal_results.get("video") is not None:
                video_result = modal_results["video"]
                video_conf = video_result.get("confidence", 0)
                st.markdown(f"- 🎥 **Video**: {video_result.get('prediction')} ({float(video_conf):.1f}%)")
            
            if modal_results.get("image") is not None:
                image_result = modal_results["image"]
                image_conf = image_result.get("confidence", 0)
                st.markdown(f"- 🖼 **Image**: {image_result.get('prediction')} ({float(image_conf):.1f}%)")
            
            if modal_results.get("audio") is not None:
                audio_result = modal_results["audio"]
                audio_conf = audio_result.get("confidence", 0)
                st.markdown(f"- 🎙 **Audio**: {audio_result.get('prediction')} ({float(audio_conf):.1f}%)")
        
        # Show GradCAM if available
        st.subheader("🔍 Explainability")
        
        video_result = result.get("modal_results", {}).get("video")
        if video_result is not None and video_result.get("gradcam_frames") is not None:
            st.write("**Video Attention Maps**")
            frames = video_result.get("gradcam_frames", [])
            cols = st.columns(3)
            for idx, frame in enumerate(frames[:6]):
                with cols[idx % 3]:
                    st.image(frame, caption=f"Frame {idx + 1}", use_container_width=True)
        
        image_result = result.get("modal_results", {}).get("image")
        if image_result is not None and image_result.get("gradcam") is not None:
            st.write("**Image Attention Map**")
            col1, col2 = st.columns(2)
            with col1:
                original = image_result.get("original")
                if original is not None:
                    st.image(original, caption="Original", use_container_width=True)
            with col2:
                gradcam = image_result.get("gradcam")
                if gradcam is not None:
                    st.image(gradcam, caption="Heatmap", use_container_width=True)

if __name__ == "__main__":
    main()
