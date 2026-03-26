"""
Fusion Inference Module

Combines per-modality predictions into a single multimodal decision.
"""

from typing import Any, Dict, Optional


DEFAULT_WEIGHTS = {
    "video": 1.0,
    "image": 1.0,
    "audio": 1.0,
}


def _clamp01(value: float) -> float:
    """Clamp floating point values to [0, 1]."""
    return max(0.0, min(1.0, value))


def _extract_fake_probability(result: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Convert a modality result dict into fake probability in [0, 1].

    Supported formats:
    1) {"probabilities": {"fake": <0..100>, "real": <0..100>}}
    2) {"prediction": "Fake"|"Real", "confidence": <0..100>}
    """
    if not result:
        return None

    probs = result.get("probabilities")
    if isinstance(probs, dict) and "fake" in probs:
        try:
            return _clamp01(float(probs["fake"]) / 100.0)
        except (TypeError, ValueError):
            return None

    prediction = str(result.get("prediction", "")).strip().lower()
    confidence = result.get("confidence", 0.0)

    try:
        conf = _clamp01(float(confidence) / 100.0)
    except (TypeError, ValueError):
        conf = 0.0

    if prediction == "fake":
        return conf
    if prediction == "real":
        return 1.0 - conf

    return None


def _normalize_weights(weights: Dict[str, float], available_modalities: Dict[str, bool]) -> Dict[str, float]:
    """Normalize weights over only available modalities."""
    filtered = {}
    for key, is_available in available_modalities.items():
        if is_available:
            filtered[key] = float(weights.get(key, 0.0))

    total = sum(filtered.values())
    if total <= 0:
        count = max(1, len(filtered))
        return {key: 1.0 / count for key in filtered}

    return {key: value / total for key, value in filtered.items()}


def fuse_predictions(
    video_result: Optional[Dict[str, Any]] = None,
    image_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Fuse modality predictions into one final decision.

    Args:
        video_result: Output dict from video_module.predict_video
        image_result: Output dict from image_module.predict_image
        audio_result: Output dict from audio module (future)
        weights: Optional modality weights (video/image/audio)
        threshold: Fake class threshold (default: 0.5)

    Returns:
        Dict with final decision and per-modality contribution.
    """
    threshold = _clamp01(threshold)

    fake_probs = {
        "video": _extract_fake_probability(video_result),
        "image": _extract_fake_probability(image_result),
        "audio": _extract_fake_probability(audio_result),
    }

    available = {key: value is not None for key, value in fake_probs.items()}
    if not any(available.values()):
        raise ValueError("No valid modality predictions were provided for fusion")

    base_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key in base_weights:
                base_weights[key] = float(value)

    normalized_weights = _normalize_weights(base_weights, available)

    fused_fake_prob = 0.0
    contributions: Dict[str, Dict[str, float]] = {}

    for modality, fake_prob in fake_probs.items():
        if fake_prob is None:
            continue

        weight = normalized_weights.get(modality, 0.0)
        weighted = fake_prob * weight
        fused_fake_prob += weighted

        contributions[modality] = {
            "fake_probability": round(fake_prob * 100.0, 2),
            "real_probability": round((1.0 - fake_prob) * 100.0, 2),
            "weight": round(weight, 4),
            "weighted_fake": round(weighted * 100.0, 2),
        }

    fused_fake_prob = _clamp01(fused_fake_prob)
    fused_real_prob = 1.0 - fused_fake_prob

    if fused_fake_prob >= threshold:
        prediction = "Fake"
        confidence = fused_fake_prob
    else:
        prediction = "Real"
        confidence = fused_real_prob

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100.0, 2),
        "threshold": round(threshold, 3),
        "probabilities": {
            "real": round(fused_real_prob * 100.0, 2),
            "fake": round(fused_fake_prob * 100.0, 2),
        },
        "contributions": contributions,
        "available_modalities": [key for key, ok in available.items() if ok],
    }


def predict_multimodal(
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
    generate_video_gradcam: bool = False,
) -> Dict[str, Any]:
    """
    Run end-to-end multimodal prediction using available module inference functions.
    """
    video_result: Optional[Dict[str, Any]] = None
    image_result: Optional[Dict[str, Any]] = None
    audio_result: Optional[Dict[str, Any]] = None

    if video_path:
        from video_module.inference import predict_video

        video_result = predict_video(video_path, generate_gradcam=generate_video_gradcam)

    if image_path:
        from image_module.inference import predict_image

        image_result = predict_image(image_path)

    if audio_path:
        from audio_module.inference import predict_audio

        audio_result = predict_audio(audio_path)

    fused = fuse_predictions(
        video_result=video_result,
        image_result=image_result,
        audio_result=audio_result,
        weights=weights,
        threshold=threshold,
    )

    fused["modal_results"] = {
        "video": video_result,
        "image": image_result,
        "audio": audio_result,
    }

    return fused
