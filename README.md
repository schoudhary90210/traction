<p align="center">
  <img src="https://img.shields.io/badge/Qualcomm-Edge_AI_Hackathon_2026-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white" alt="Qualcomm Edge AI Hackathon 2026"/>
  <img src="https://img.shields.io/badge/Snapdragon_X_Elite-NPU_45_TOPS-E31937?style=for-the-badge" alt="Snapdragon X Elite"/>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.17+-7B1FA2?style=for-the-badge" alt="ONNX Runtime"/>
</p>

# TRACTION

**Real-Time Crop Disease Detection System for Edge AI**

Traction is a tractor-mounted crop disease detection dashboard built for the **Qualcomm Edge AI Hackathon 2026**. It uses a ConvNeXtV2 vision model running on ONNX Runtime with Snapdragon X Elite NPU acceleration, a GPS-enabled ghost tractor simulator with Google Maps heatmaps, and an offline LLM treatment advisor powered by Ollama — all in a single Streamlit interface designed for in-cab operation.

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Dashboard Tabs](#dashboard-tabs)
- [Vision Engine](#vision-engine)
- [Map Engine & GPS Simulation](#map-engine--gps-simulation)
- [Offline AI Treatment Advisor](#offline-ai-treatment-advisor)
- [Training Pipeline](#training-pipeline)
- [ONNX Export](#onnx-export)
- [Disease Detection Log Schema](#disease-detection-log-schema)
- [Configuration Reference](#configuration-reference)
- [Threading Architecture](#threading-architecture)
- [Dependencies](#dependencies)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Team](#team)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/schoudhary90210/traction.git
cd traction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Click **START** on the camera feed to begin real-time inference.

### Optional: Enable AI Treatment Plans

```bash
# Install Ollama (https://ollama.ai/download)
ollama serve                    # Start the server
ollama pull mistral             # Download the model (~7GB, one-time)
```

Traction auto-detects Ollama on startup. If unavailable, static USDA-guideline-based fallback advice is used.

---

## System Architecture

```
Tractor Dashcam (30 FPS)
       |
       v
 WebRTC Stream (streamlit-webrtc)
       |
       v  (every 5th frame)
 _frame_slot [thread-safe buffer]
       |
       v
 Inference Worker Thread
       |
       v
 TractionEngine.process_frame()
   +-- ONNX Runtime (QNN > CUDA > CPU)
   +-- ConvNeXtV2-Atto (agri_convnext.onnx)
   +-- 15-frame majority-vote temporal smoothing
       |
       v
 _result_* globals [lock-protected]
       |
       +---> Treatment Worker Thread
       |        +-- Ollama API (POST /api/generate)
       |        +-- Mistral model, temp=0.3
       |        +-- Fallback: static treatment dict
       |        +-- Result -> st.session_state.advisory_text
       |
       +---> Disease Logger
       |        +-- Confidence threshold (0.58)
       |        +-- Time debounce (5.0s)
       |        +-- Distance debounce (8.0m, haversine)
       |        +-- Writes to disease_detections.json
       |
       v
 Streamlit Main Thread
   +-- Session state sync (lock -> session_state)
   +-- 7-tab dashboard render
   +-- Google Maps HTML (embedded via components.html)
```

---

## Project Structure

```
traction/
+-- app.py                      # Main Streamlit dashboard (2,500+ lines)
+-- vision_engine.py            # ONNX inference + temporal smoothing
+-- map_engine.py               # Tractor simulator + Google Maps renderer
+-- agri_convnext.onnx          # ConvNeXtV2-Atto model (primary)
+-- agri_convnext.pth           # PyTorch weights
+-- disease_detections.json     # Persisted geo-tagged detection log
+-- requirements.txt            # All dependencies
+-- README.md                   # This file
+-- train_model.py              # MobileNetV2 training pipeline
+-- convert_to_onnx.py          # PyTorch -> ONNX export (frozen weight fix)
+-- test_inference.py           # Local ONNX inference test utility
+-- models/
|   +-- agri_scout_mobilenetv2.onnx   # Fallback model (~9 MB)
+-- dataset/
|   +-- archive/                # PlantVillage corn disease dataset (augmented)
+-- tests/
    +-- HEATMAP_MANUAL_TESTS.md # Manual QA checklist
```

---

## Dashboard Tabs

### 1. Drive (Main Tab)

The primary operator view for in-cab use.

| Component | Description |
|-----------|-------------|
| **Operator Alert Bar** | Color-coded status: SCANNING (blue), ALL CLEAR (green), ACTION NEEDED (red) |
| **Live Camera Feed** | WebRTC stream with OSD overlay (label, confidence, latency, frame border) |
| **Detection Panel** | Current disease classification with confidence bar |
| **System Metrics** | Inference count, smoothing window, detection mode |
| **AI Treatment Advisory** | Ollama-generated or static treatment plan |
| **Generate Treatment Button** | Manual trigger for treatment plan from latest logged detection |
| **Ghost Tractor Map** | Google Maps with tractor position, path trail, disease markers, heatmap |
| **Map Controls** | Precision pan/zoom, basemap type, coordinate input |

### 2. Data Visualization

Analytics dashboard with synthetic and real disease data.

- Configurable sample size (120-1,200 points), heatmap opacity/radius
- Disease distribution charts (bar, pie, average alert by disease)
- Risk bands, environmental averages, alert histograms, top zones by detection density
- Interactive Google Maps heatmap with marker threshold filtering

### 3. Logs + GPS

Raw detection history and GPS telemetry.

- Current GPS fix (latitude, longitude, heading, cardinal direction)
- Log entry count from `disease_detections.json`
- Scrollable table of last 100 detections (timestamp, disease, confidence %, lat, lng)

### 4. NPU/GPU/CPU Power

Hardware compute telemetry.

- CPU, GPU, NPU utilization gauges
- NPU efficiency ratio
- Runtime provider info from `TractionEngine.get_runtime_info()`

### 5. Gallery

Media browser for captured images and dataset samples.

- Filter by type (image/video) and source (recordings, dataset)
- Thumbnail grid with metadata (name, kind, size, folder)
- Configurable item limit (12-96)

### 6. Settings

Full configuration panel.

- **Display**: Night/Day/Emergency presets, 4 color schemes, high contrast, color inversion
- **Farm Location**: Name, latitude, longitude, field width/height, disease spot size
- **Profiles**: Save/load/delete farm configurations with map type and spot size

### 7. Help + Demo

Usage guide and demo video player.

---

## Vision Engine

**File:** `vision_engine.py`
**Class:** `TractionEngine`

### Model Priority

```
1. agri_convnext.onnx       (project root, ConvNeXtV2-Atto, primary)
2. models/agri_scout_mobilenetv2.onnx  (fallback)
3. Mock mode                 (no model file found, random predictions)
```

### Execution Provider Priority

```
1. QNNExecutionProvider      (Snapdragon X Elite NPU, 45 TOPS)
2. CUDAExecutionProvider     (NVIDIA GPU)
3. CPUExecutionProvider      (fallback)
```

### Disease Classes

| Index | Label |
|-------|-------|
| 0 | Healthy |
| 1 | Northern Leaf Blight |
| 2 | Cercospora (Gray Leaf Spot) |
| 3 | Common Rust |

### Preprocessing Pipeline

```
Input: BGR frame (e.g., 1920x1080)
  |
  +-- BGR -> RGB
  +-- Center crop: square of min(H,W)/2  (e.g., 540x540 from 1080p)
  +-- Resize to 224x224 (bilinear)
  +-- Normalize to [0,1], subtract ImageNet mean, divide by ImageNet std
  +-- HWC -> CHW transpose
  +-- Add batch dimension -> (1, 3, 224, 224)
```

### Temporal Smoothing

**Problem:** Raw per-frame predictions flicker due to camera noise, motion blur, and sunlight glare.

**Solution:** Majority voting over a 15-frame sliding window (~2-3 seconds at ~5 FPS inference rate).

```
History buffer (deque, maxlen=15):
  [("Blight", 0.91), ("Blight", 0.87), ("Rust", 0.72),
   ("Blight", 0.93), ("Blight", 0.85), ("Rust", 0.68),
   ("Blight", 0.90)]

Counter: {Blight: 5, Rust: 2}
Winner:  "Blight"
Avg confidence (Blight only): (0.91+0.87+0.93+0.85+0.90)/5 = 0.892
Output: ("Blight", 0.892)
```

- Odd window size (15) prevents ties
- Only averages confidence of the winning class
- Auto-evicts oldest entry on append

### Public API

```python
engine = TractionEngine()                          # Auto-selects model + provider
label, confidence = engine.process_frame(bgr_img)  # Smoothed prediction
engine.reset_history()                             # Clear buffer on scene change
info = engine.get_runtime_info()                   # Runtime diagnostics dict
```

### `get_runtime_info()` Return Value

```python
{
    "runtime_provider": "QNNExecutionProvider",
    "available_providers": ["QNNExecutionProvider", "CPUExecutionProvider"],
    "model_path": "/path/to/agri_convnext.onnx",
    "window_size": 15,
    "mock_mode": False,
}
```

---

## Map Engine & GPS Simulation

**File:** `map_engine.py`

### TractorSimulator

Simulates a tractor driving a boustrophedon (lawnmower) grid pattern across a configurable rectangular field.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_lat` | 43.1580 | Starting latitude (Arena, Wisconsin) |
| `start_lng` | -89.9120 | Starting longitude |
| `speed_mph` | 5.0 | Ground speed (~2.24 m/s) |
| `field_width_m` | 800.0 | East-west extent |
| `field_height_m` | 708.0 | North-south extent (~140 acres) |
| `row_spacing_m` | 12.0 | Spacing between passes |

**Movement pattern:**
- Even rows: West to East (heading 90 degrees)
- Odd rows: East to West (heading 270 degrees)
- Transitions: South (heading 180 degrees)
- Position advances by wall-clock elapsed time

**Path history:** Stores last 400 waypoints for trail rendering.

### DiseaseLog Schema

```python
@dataclass
class DiseaseLog:
    lat: float          # Latitude
    lng: float          # Longitude
    label: str          # Disease name
    confidence: float   # 0.0 - 1.0
    timestamp: float    # Unix timestamp
```

### Google Maps Renderer

`generate_map_html()` returns self-contained HTML/JS with:

| Feature | Details |
|---------|---------|
| **Tractor marker** | Green forward arrow, rotates with heading, z-index 999 |
| **Path trail** | Green polyline, last 300 waypoints, opacity 0.35 |
| **Field boundary** | Green rectangle overlay, opacity 0.15 |
| **Disease markers** | Red circles with halos, click for info popup |
| **Heatmap layer** | Weighted by confidence, blue-to-red gradient |
| **Dark mode** | Desaturated map style, `#08090c` background |
| **Safety** | Coordinates clamped to web-mercator-safe ranges |

Supports 4 basemap types: `roadmap`, `hybrid`, `satellite`, `terrain`.

---

## Offline AI Treatment Advisor

### Architecture

```
Disease detected (conf >= threshold)
  |
  v
Treatment Worker Thread (daemon)
  |
  v
POST http://localhost:11434/api/generate
  {
    "model": "mistral",
    "prompt": "<agricultural expert prompt>",
    "stream": false,
    "options": {
      "temperature": 0.3,
      "top_k": 40,
      "top_p": 0.9,
      "num_predict": 150
    }
  }
  |
  +-- Success: HTML-formatted treatment plan
  +-- Failure: Static fallback advice
  |
  v
st.session_state.advisory_text -> Advisory Panel
```

### Prompt Template

The LLM receives:
- Disease name and confidence percentage
- GPS coordinates (if available)
- Last 10 detection entries from `disease_detections.json` for field context

### Static Fallback Treatments

| Disease | Treatment |
|---------|-----------|
| Common Rust | Copper/sulfur fungicide every 7-10 days |
| Northern Leaf Blight | Chlorothalonil/propiconazole within 48 hours |
| Cercospora / Gray Leaf Spot | Mancozeb/azoxystrobin immediately |
| Generic | Contact local extension agent |

### Performance

| Metric | Value |
|--------|-------|
| Latency | ~1-2 seconds per plan (warm), ~10-20 seconds (cold) |
| Model size | ~7 GB (Mistral) |
| Connectivity | Fully offline after model download |
| Cost | Free and open-source |

---

## Training Pipeline

**File:** `train_model.py`

### Dataset

PlantVillage Corn Disease Dataset (Augmented):

| Class | Folder Name |
|-------|-------------|
| Cercospora (Gray Leaf Spot) | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| Common Rust | `Corn_(maize)___Common_rust_` |
| Northern Leaf Blight | `Corn_(maize)___Northern_Leaf_Blight` |
| Healthy | `Corn_(maize)___healthy` |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 (ImageNet pretrained) |
| Classifier head | Dropout(0.2) -> Linear(1280, 4) |
| Backbone | Frozen (transfer learning) |
| Optimizer | Adam, lr=1e-3 |
| Loss | CrossEntropyLoss |
| Epochs | 5 |
| Batch size | 32 |
| Input size | 224x224 |
| ONNX opset | 14 (Qualcomm AI Hub compatible) |

### Data Augmentation

```
Training:  Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip(0.5)
           -> RandomRotation(15) -> GaussianBlur(7, sigma=0.1-2.0)
           -> ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05)
           -> Normalize(ImageNet)

Validation: Resize(256) -> CenterCrop(224) -> Normalize(ImageNet)
```

GaussianBlur and ColorJitter simulate real tractor dashcam conditions (motion blur, sunlight glare, exposure changes).

---

## ONNX Export

**File:** `convert_to_onnx.py`

### Problem Solved

PyTorch 2.x Dynamo exporter drops frozen (requires_grad=False) backbone parameters, producing a 0.26 MB head-only file instead of the full 9+ MB model.

### Solution

1. Reconstruct exact MobileNetV2 architecture
2. Load trained weights from `.pth`
3. **Unfreeze all parameters** (set `requires_grad=True`)
4. Force legacy JIT tracer (`dynamo=False`)
5. Validate output size > 8 MB

```python
# The fix: unfreeze before export
for param in model.parameters():
    if not param.requires_grad:
        param.requires_grad = True
```

| Export Setting | Value |
|----------------|-------|
| Opset version | 14 |
| Constant folding | Enabled |
| Dynamic axes | Batch dimension |
| Input names | `["input"]` |
| Output names | `["output"]` |
| Expected size | 9-14 MB |

---

## Disease Detection Log Schema

**File:** `disease_detections.json`

```json
[
  {
    "timestamp": "2026-02-21T20:19:28.384516",
    "disease": "Cercospora (Gray Leaf Spot)",
    "confidence": 0.5797,
    "lat": 43.158000,
    "lng": -89.911413
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string (ISO 8601) | Detection time (`YYYY-MM-DDTHH:MM:SS.ffffff`) |
| `disease` | string | Disease class label |
| `confidence` | float | Model confidence (0.0 - 1.0) |
| `lat` | float | Latitude (6 decimal places) |
| `lng` | float | Longitude (6 decimal places) |

On startup, `_merge_benji_detections_into_session()` loads this file, filters entries below 0.40 confidence, converts timestamps to unix floats, deduplicates, and pre-populates the map with historical detections.

---

## Configuration Reference

### Detection Thresholds

| Constant | Value | Purpose |
|----------|-------|---------|
| `_DISEASE_CONFIDENCE_THRESH` | 0.85 | Alert threshold (operator warning) |
| `_LOG_CONFIDENCE_THRESH` | 0.58 | Logging threshold (write to JSON) |
| `_LOG_COOLDOWN_SEC` | 5.0 | Time debounce between logs |
| `_LOG_MIN_DISTANCE_M` | 8.0 | Spatial debounce (haversine) |
| `_INFERENCE_COOLDOWN` | 0.15 | Sleep between inferences |
| `_FRAME_SKIP` | 5 | Process every Nth WebRTC frame |
| `_OUTPUT_LOG_INTERVAL_SEC` | 1.0 | JSON write interval |
| `SMOOTHING_WINDOW` | 15 | Temporal smoothing frame count |

### Ollama

| Constant | Value |
|----------|-------|
| `_OLLAMA_BASE_URL` | `http://localhost:11434` |
| `_OLLAMA_MODEL` | `mistral` |
| `_OLLAMA_TIMEOUT` | 15 seconds |

### Farm Defaults

| Constant | Value |
|----------|-------|
| `_DEFAULT_FARM_NAME` | `"North-40"` |
| `_DEFAULT_FARM_LAT` | `43.1580` (Arena, Wisconsin) |
| `_DEFAULT_FARM_LNG` | `-89.9120` |
| `_DEFAULT_FARM_WIDTH_M` | `800.0` |
| `_DEFAULT_FARM_HEIGHT_M` | `708.0` (~140 acres) |
| `_DEFAULT_DISEASE_SPOT_SIZE` | `10.0` |

### Color Schemes

| Scheme | Background | Text | Use Case |
|--------|-----------|------|----------|
| Night Field (Dark) | `#08090c` | `#eaedf3` | Default, low-light cab |
| Daylight Paper | `#fffdf4` | `#101010` | Bright sunlight |
| Green Cabin | `#07110d` | `#f0fff6` | Dark green tint |
| Amber Contrast | `#2a1e00` | `#0f0f0f` | Emergency / high visibility |

---

## Threading Architecture

Traction runs 3 concurrent threads to keep the UI responsive:

| Thread | Name | Purpose | Wake Mechanism |
|--------|------|---------|----------------|
| **Inference Worker** | `traction-infer` | ONNX model inference | `_frame_event` (Event) |
| **Treatment Worker** | `traction-treatment` | Ollama LLM calls | `_treatment_event` (Event) |
| **Main Thread** | Streamlit | UI rendering, session state | Streamlit rerun cycle |

### Thread-Safe Globals

```
_frame_lock    -> protects _frame_slot (frame buffer)
_result_lock   -> protects _result_label, _result_conf, _result_latency_ms
_treatment_lock -> protects _treatment_pending_*, _treatment_result_*
_output_log_lock -> protects JSON write timing
```

### Synchronization Flow

1. WebRTC callback fires at 30 FPS, copies every 5th frame to `_frame_slot`
2. Inference worker wakes on `_frame_event`, runs model, stores result under `_result_lock`
3. If disease detected and new label, sets `_treatment_event`
4. Treatment worker wakes, calls Ollama, stores plan under `_treatment_lock`
5. Main Streamlit thread reads all locks, syncs to `st.session_state`, renders UI

---

## Dependencies

```
streamlit>=1.32.0              # Dashboard framework
streamlit-webrtc>=0.47.0       # Real-time camera via WebRTC
streamlit-autorefresh>=1.0.1   # Auto-refresh component
opencv-python-headless>=4.8.0  # Image preprocessing (no GUI)
numpy>=1.24.0                  # Array operations
onnxruntime>=1.17.0            # Model inference (QNN/CUDA/CPU)
av>=12.0.0                     # Video frame handling (PyAV)
requests>=2.28.0               # Ollama HTTP API client
psutil>=5.9.0                  # System telemetry
pandas>=2.0.0                  # Data tables
```

---

## Deployment

### Hardware Target

| Component | Spec |
|-----------|------|
| SoC | Snapdragon X Elite |
| NPU | Qualcomm Hexagon, 45 TOPS |
| Quantization | INT8 supported |
| QNN SDK | 2.28+ |
| ONNX Opset | 14 |

### Compile for NPU

```bash
# Using Qualcomm AI Hub CLI
qai-hub compile agri_convnext.onnx --device "Snapdragon X Elite"
```

### Production Checklist

- [ ] Replace `GOOGLE_MAPS_API_KEY` with production key
- [ ] Install Ollama and pull mistral model for treatment AI
- [ ] Connect USB camera or dashcam (MJPEG/H.264)
- [ ] Test with `QNNExecutionProvider` on target hardware
- [ ] Verify GPS coordinates match actual farm location in Settings

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not showing | Check browser WebRTC permissions, try Chrome/Edge |
| Model not loading | Verify `agri_convnext.onnx` exists in project root |
| Map not rendering | Check `GOOGLE_MAPS_API_KEY` is valid and has Maps JS API enabled |
| Ollama not found | Run `ollama serve` in a separate terminal |
| Slow first treatment | First Ollama call loads model into RAM (~10-20s), subsequent calls are fast |
| Predictions flickering | Normal during first 15 frames while smoothing buffer fills |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## Team

Built for the **Qualcomm Edge AI Hackathon 2026** by:

- **Christian** — Full Streamlit UI, 7-tab dashboard, enterprise CSS, Google Maps integration, farm profiles, color schemes
- **Benji** — Real GPS disease detection data, Ollama LLM treatment advisor, detection logging, Logs+GPS tab
- **Sid** — ConvNeXtV2-Atto model training, ONNX export pipeline, vision engine with temporal smoothing

---

<p align="center">
  <sub>Qualcomm Edge AI Hackathon 2026</sub>
</p>
