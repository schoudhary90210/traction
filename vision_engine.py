#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Vision Engine (ONNX Runtime Inference Backend)
================================================================================
 v2 — Temporal Smoothing via Majority Voting
--------------------------------------------------------------------------------
 Problem:  Raw per-frame predictions flicker wildly due to camera noise, motion
           blur, and sunlight glare on the tractor dashcam.
 Fix:      Maintain a sliding window of the last N predictions (default 7).
           Return the majority-vote label + averaged confidence of that class,
           giving the dashboard a stable, "locked-in" result.
================================================================================
 API contract (unchanged — drop-in replacement):
     engine = AgriScoutEngine("models/agri_scout_mobilenetv2.onnx")
     label, confidence = engine.process_frame(frame_bgr)
================================================================================
"""

import os
from collections import Counter, deque

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ==============================================================================
#  Constants
# ==============================================================================

# Must match the order produced by ImageFolder (alphabetical sort)
CLASS_LABELS = [
    "Cercospora (Gray Leaf Spot)",
    "Common Rust",
    "Northern Leaf Blight",
    "Healthy",
]

# ImageNet normalisation used during training
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = os.path.join("models", "agri_scout_mobilenetv2.onnx")

# Temporal smoothing window size.
# 7 frames ≈ 1–2 seconds at the async pipeline's ~5 FPS inference rate.
# Odd number ensures a clear majority winner with no ties.
# Change this from 7 to 15
SMOOTHING_WINDOW = 15

# ==============================================================================
#  AgriScoutEngine
# ==============================================================================

class AgriScoutEngine:
    """
    ONNX inference wrapper with temporal smoothing (majority voting).

    Each call to process_frame() runs the model on the current frame,
    appends the raw (label, confidence) to a sliding deque of size
    SMOOTHING_WINDOW, then returns the majority-vote label and the
    averaged confidence of all matching predictions in the window.

    Usage:
        engine = AgriScoutEngine("models/agri_scout_mobilenetv2.onnx")
        label, confidence = engine.process_frame(frame_bgr)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.session = None

        # -- Temporal smoothing history --
        # Stores tuples of (label_str, confidence_float) for the last N frames.
        # maxlen=SMOOTHING_WINDOW auto-evicts the oldest entry on append.
        self.history: deque = deque(maxlen=SMOOTHING_WINDOW)

        if ort is None:
            print("[ENGINE] [WARN] onnxruntime not installed -- running in MOCK mode.")
            print("         Install with:  pip install onnxruntime")
            return

        if not os.path.isfile(model_path):
            print(f"[ENGINE] [WARN] Model not found at '{model_path}' -- running in MOCK mode.")
            return

        # Prefer CPU EP for broadest compatibility; swap to QNN EP on-device.
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        meta = self.session.get_inputs()[0]
        print(f"[ENGINE] [OK]  Model loaded: {model_path}")
        print(f"[ENGINE]    Input : {meta.name}  shape={meta.shape}  dtype={meta.type}")
        print(f"[ENGINE]    Output: {self.session.get_outputs()[0].name}")
        print(f"[ENGINE]    Temporal smoothing: majority vote over {SMOOTHING_WINDOW} frames")

# ------------------------------------------------------------------
    #  Pre-processing (Perfect 1:1 UI Reticle Mapping)
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        """
        Crops the EXACT square where the UI reticle is drawn on screen,
        then resizes that exact crop for the AI to ensure 1:1 targeting.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # 1. Define the exact same physical box size as the app.py UI
        box_size = min(h, w) // 2
        y0 = (h - box_size) // 2
        x0 = (w - box_size) // 2

        # 2. Crop that exact square out of the high-res 1080p image
        crop = rgb[y0 : y0 + box_size, x0 : x0 + box_size]

        # 3. Resize that perfect crop down to the 224x224 the AI expects
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 4. Standard ImageNet normalization
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD

        # 5. HWC → CHW, add batch dim → (1, 3, 224, 224)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob
    # ------------------------------------------------------------------
    #  Raw single-frame inference (internal)
    # ------------------------------------------------------------------
    def _infer_single(self, frame_bgr: np.ndarray) -> tuple:
        """Run the ONNX model on one frame. Returns (label, confidence)."""
        blob = self._preprocess(frame_bgr)
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: blob})[0]  # (1, 4)

        # Softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(probs[0, idx])

        return CLASS_LABELS[idx], conf

    # ------------------------------------------------------------------
    #  Temporal smoothing (majority vote over sliding window)
    # ------------------------------------------------------------------
    def _smooth(self) -> tuple:
        """
        Compute the majority-vote label from self.history and return
        (winning_label, averaged_confidence_of_that_label).

        Example with history = [
            ("Blight", 0.91), ("Blight", 0.87), ("Rust", 0.72),
            ("Blight", 0.93), ("Blight", 0.85), ("Rust", 0.68),
            ("Blight", 0.90)
        ]:
            → Counter: Blight=5, Rust=2
            → Winner:  "Blight"
            → Avg conf of Blight entries: (0.91+0.87+0.93+0.85+0.90)/5 = 0.892
            → Returns: ("Blight", 0.892)
        """
        if not self.history:
            return "Waiting…", 0.0

        # Count label occurrences
        label_counts = Counter(label for label, _ in self.history)
        winner_label, _ = label_counts.most_common(1)[0]

        # Average confidence only for frames that predicted the winning class
        matching_confs = [conf for label, conf in self.history if label == winner_label]
        avg_conf = sum(matching_confs) / len(matching_confs)

        return winner_label, round(avg_conf, 4)

    # ------------------------------------------------------------------
    #  Public API — process_frame  (smoothed output)
    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> tuple:
        """
        Run inference on a single BGR frame, then return the temporally
        smoothed (majority-vote) prediction.

        Returns:
            (predicted_class_string, smoothed_confidence_float)
            e.g. ("Northern Leaf Blight", 0.8921)
        """
        # --- Mock mode (no model loaded) ---
        if self.session is None:
            raw_label, raw_conf = self._mock_inference()
        else:
            raw_label, raw_conf = self._infer_single(frame_bgr)

        # Append raw prediction to sliding window
        self.history.append((raw_label, raw_conf))

        # Return the smoothed (majority-vote) result
        return self._smooth()

    # ------------------------------------------------------------------
    #  Utility: reset the smoothing buffer
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Clear the prediction history. Useful on scene/field changes."""
        self.history.clear()

    # ------------------------------------------------------------------
    #  Mock fallback (for UI development without the model file)
    # ------------------------------------------------------------------
    @staticmethod
    def _mock_inference() -> tuple:
        """Return a random but realistic-looking prediction for UI testing."""
        idx = np.random.choice(len(CLASS_LABELS), p=[0.25, 0.35, 0.20, 0.20])
        conf = np.random.uniform(0.78, 0.99)
        return CLASS_LABELS[idx], round(conf, 4)