#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Real-Time Crop Disease Detection Dashboard  (v4 — Ghost Tractor)
================================================================================
 Run with:   streamlit run app.py
================================================================================
"""

import datetime
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from vision_engine import AgriScoutEngine

# ==============================================================================
#  Safe import of map_engine (graceful fallback if file is missing)
# ==============================================================================

_MAP_AVAILABLE = False
TractorSimulator = None
DiseaseLog = None
generate_map_html = None
try:
    from map_engine import TractorSimulator, DiseaseLog, generate_map_html
    _MAP_AVAILABLE = True
    _DEFAULT_FARM_LAT = TractorSimulator.DEFAULT_START_LAT
    _DEFAULT_FARM_LNG = TractorSimulator.DEFAULT_START_LNG
    _DEFAULT_FARM_WIDTH_M = TractorSimulator.FIELD_WIDTH_M
    _DEFAULT_FARM_HEIGHT_M = TractorSimulator.FIELD_HEIGHT_M
except Exception as _map_import_err:
    # If map_engine.py is missing or broken, the rest of the app still works.
    # We'll show a clear error message in the map section instead.
    pass


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

GOOGLE_MAPS_API_KEY = "AIzaSyAw2m1LGnhQ0C4jkyN5Z2aH5ADTPGfQZZ4"

_LOG_COOLDOWN_SEC = 5.0
_LOG_MIN_DISTANCE_M = 8.0
_DISEASE_CONFIDENCE_THRESH = 0.85
_COLOR_SCHEMES = [
    "Night Field (Dark)",
    "Daylight Paper (Black on Light)",
    "Green Cabin (White on Dark Green)",
    "Amber Contrast (Black on Amber)",
]
_DISPLAY_PRESETS = ["Night", "Day", "Emergency"]
_DEFAULT_FARM_NAME = "North-40"
_DEFAULT_FARM_LAT = globals().get("_DEFAULT_FARM_LAT", 43.1580)
_DEFAULT_FARM_LNG = globals().get("_DEFAULT_FARM_LNG", -89.9120)
_DEFAULT_FARM_WIDTH_M = globals().get("_DEFAULT_FARM_WIDTH_M", 800.0)
_DEFAULT_FARM_HEIGHT_M = globals().get("_DEFAULT_FARM_HEIGHT_M", 708.0)
_DEFAULT_DISEASE_SPOT_SIZE = 10.0
_MAP_TYPES = ["roadmap", "hybrid", "satellite", "terrain"]


# ==============================================================================
#  Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="Agri-Scout | Qualcomm Edge AI",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
#  Enterprise CSS
# ==============================================================================

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;-webkit-font-smoothing:antialiased}
code,pre,.mono{font-family:'IBM Plex Mono',monospace !important}
.stApp{background:#08090c}
div[data-testid="stAppViewBlockContainer"]{padding-top:1rem;padding-bottom:1rem;max-width:100%}
header[data-testid="stHeader"]{background:transparent}
div[data-testid="stToolbar"]{display:flex !important}
[data-testid="collapsedControl"]{
  display:flex !important;
  visibility:visible !important;
  position:fixed;
  top:0.6rem;
  left:0.6rem;
  z-index:1000;
  background:#111820cc;
  border:1px solid #2b3448;
  border-radius:8px;
  padding:2px 4px;
}
[data-testid="stDecoration"]{display:none}
section[data-testid="stSidebar"]{background:#0c0e14;border-right:1px solid #1a1d28}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#a0a4b0}
.hdr{background:linear-gradient(135deg,#0d1117,#111820);border:1px solid #1a1f2e;border-radius:14px;padding:1rem 1.6rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between}
.hdr.hdr-compact{padding:0.65rem 1rem;margin-bottom:0.7rem}
.hdr-left{display:flex;align-items:center;gap:14px}
.hdr-logo{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#00e676,#00bfa5);display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:700;color:#08090c;flex-shrink:0}
.hdr h1{font-size:1.35rem;font-weight:700;color:#eaedf3;margin:0;letter-spacing:-0.3px;line-height:1.2}
.hdr .sub{font-size:0.72rem;color:#5a6072;text-transform:uppercase;letter-spacing:1.5px;margin-top:1px}
.hdr-right{display:flex;align-items:center;gap:10px}
.hdr-chip{padding:0.28rem 0.75rem;border-radius:6px;font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px}
.chip-live{background:rgba(0,230,118,0.12);color:#00e676;border:1px solid rgba(0,230,118,0.25)}
.chip-npu{background:rgba(100,120,255,0.10);color:#8890ff;border:1px solid rgba(100,120,255,0.20)}
.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle}
.live-dot.green{background:#00e676;box-shadow:0 0 6px #00e67688;animation:blink 2s infinite}
.live-dot.red{background:#ff3d3d;box-shadow:0 0 6px #ff3d3d88;animation:blink 1.5s infinite}
.live-dot.blue{background:#6478ff;box-shadow:0 0 6px #6478ff88;animation:blink 2.5s infinite}
.live-dot.amber{background:#ffab00;box-shadow:0 0 6px #ffab0088;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.sidebar-section-label{font-size:0.62rem;color:#4a4e5c;text-transform:uppercase;letter-spacing:2px;margin:1rem 0 0.5rem 0;font-weight:600}
.tcard{background:#10121a;border:1px solid #1a1d28;border-radius:10px;padding:0.7rem 0.85rem;margin-bottom:0.45rem;transition:border-color 0.2s}
.tcard:hover{border-color:#2a3040}
.tcard .tc-label{font-size:0.62rem;color:#4a5060;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:0.15rem;font-weight:500}
.tcard .tc-val{font-size:1.15rem;font-weight:600;color:#e0e3eb;font-family:'IBM Plex Mono',monospace;line-height:1.25}
.tcard .tc-unit{font-size:0.7rem;color:#3d9970;font-weight:500;vertical-align:baseline;position:relative;top:-1px;margin-left:4px}
.tcard .tc-sub{font-size:0.65rem;color:#4a5060;margin-top:2px}
.feed-wrap{background:#0c0e14;border:1px solid #1a1d28;border-radius:14px;padding:0.6rem;margin-bottom:0.8rem}
.feed-label{font-size:0.6rem;color:#4a5060;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;padding-left:0.3rem;font-weight:600}
.det-panel{border-radius:12px;padding:1rem 1.3rem;margin-bottom:0.8rem;display:flex;align-items:center;gap:16px}
.det-panel.healthy{background:linear-gradient(135deg,#0a1f12,#0d2818);border:1px solid #1a4028}
.det-panel.disease{background:linear-gradient(135deg,#1f0a0a,#281010);border:1px solid #4a1a1a}
.det-panel.waiting{background:#10121a;border:1px solid #1a1d28}
.det-icon{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0}
.det-icon.healthy{background:rgba(0,230,118,0.12)}
.det-icon.disease{background:rgba(255,61,61,0.12)}
.det-icon.waiting{background:rgba(100,120,255,0.10)}
.det-body{flex:1}
.det-class{font-size:1.15rem;font-weight:700;color:#eaedf3;margin-bottom:2px}
.det-meta{font-size:0.72rem;color:#5a6072;font-family:'IBM Plex Mono',monospace}
.det-conf-bar{width:100%;max-width:180px;height:6px;background:#1a1d28;border-radius:3px;margin-top:6px;overflow:hidden}
.det-conf-fill{height:100%;border-radius:3px;transition:width 0.3s ease}
.det-conf-fill.healthy{background:linear-gradient(90deg,#00e676,#00bfa5)}
.det-conf-fill.disease{background:linear-gradient(90deg,#ff3d3d,#ff6b6b)}
.advisory-panel{background:#0c0e14;border:1px solid #1a1d28;border-radius:14px;padding:1.2rem 1.4rem;margin-top:0.2rem;position:relative;overflow:hidden}
.advisory-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6478ff,#00e676,#6478ff);opacity:0.6}
.adv-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.9rem}
.adv-header-left{display:flex;align-items:center;gap:8px}
.adv-icon{width:28px;height:28px;border-radius:8px;background:rgba(100,120,255,0.10);border:1px solid rgba(100,120,255,0.18);display:flex;align-items:center;justify-content:center;font-size:0.85rem}
.adv-title{font-size:0.68rem;color:#8890ff;text-transform:uppercase;letter-spacing:1.8px;font-weight:600}
.adv-status{font-size:0.6rem;color:#737c92;text-transform:uppercase;letter-spacing:1px;font-family:'IBM Plex Mono',monospace;font-weight:500}
.adv-body{background:#0a0c12;border:1px solid #141722;border-radius:10px;padding:1rem 1.1rem;min-height:100px}
.adv-placeholder{color:#a5adbf;font-size:0.84rem;font-family:'IBM Plex Mono',monospace;line-height:1.65}
.adv-placeholder .cursor{display:inline-block;width:2px;height:14px;background:#6478ff;margin-left:3px;vertical-align:text-bottom;animation:cursor-blink 1s steps(2) infinite}
@keyframes cursor-blink{0%{opacity:1}100%{opacity:0}}
.adv-content{color:#c8ccd6;font-size:0.85rem;line-height:1.7}
.adv-content strong{color:#eaedf3}
.adv-footer{display:flex;align-items:center;gap:16px;margin-top:0.8rem;padding-top:0.7rem;border-top:1px solid #141722}
.adv-footer-item{font-size:0.6rem;color:#3d4250;text-transform:uppercase;letter-spacing:1px;font-family:'IBM Plex Mono',monospace}
.map-panel{background:#0c0e14;border:1px solid #1a1d28;border-radius:14px;padding:1rem 1.2rem;margin-top:0.8rem;position:relative;overflow:hidden}
.map-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00e676,#00bfa5,#00e676);opacity:0.5}
.map-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.7rem}
.map-header-left{display:flex;align-items:center;gap:8px}
.map-icon{width:28px;height:28px;border-radius:8px;background:rgba(0,230,118,0.10);border:1px solid rgba(0,230,118,0.18);display:flex;align-items:center;justify-content:center;font-size:0.85rem}
.map-title{font-size:0.68rem;color:#00e676;text-transform:uppercase;letter-spacing:1.8px;font-weight:600}
.map-stats{display:flex;gap:16px}
.map-stat{font-size:0.6rem;color:#3d4250;text-transform:uppercase;letter-spacing:1px;font-family:'IBM Plex Mono',monospace}
.map-stat strong{color:#ff6b6b;font-weight:600}
.metrics-row{display:flex;gap:10px;margin-top:0.6rem}
.mcard{flex:1;background:#10121a;border:1px solid #1a1d28;border-radius:10px;padding:0.65rem 0.85rem;text-align:center}
.mcard .mc-label{font-size:0.58rem;color:#4a5060;text-transform:uppercase;letter-spacing:1.5px;font-weight:500}
.mcard .mc-val{font-size:0.95rem;font-weight:600;color:#c8ccd6;font-family:'IBM Plex Mono',monospace;margin-top:2px}
.cab-alert{border-radius:14px;padding:1rem 1.1rem;margin-bottom:0.8rem;border:2px solid}
.cab-alert.safe{background:linear-gradient(135deg,#082012,#0d2a18);border-color:#1f6f44}
.cab-alert.warn{background:linear-gradient(135deg,#261900,#332000);border-color:#8c6a00}
.cab-alert.danger{background:linear-gradient(135deg,#2a0d0d,#351414);border-color:#9b2d2d}
.cab-alert-top{display:flex;align-items:center;justify-content:space-between;gap:12px}
.cab-alert-state{font-size:1.05rem;font-weight:800;letter-spacing:0.5px;color:#f2f4f8;text-transform:uppercase}
.cab-alert-meta{font-size:0.8rem;color:#b4bac7;font-family:'IBM Plex Mono',monospace;margin-top:4px}
.cab-pill{padding:0.32rem 0.66rem;border-radius:999px;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.9px}
.cab-pill.safe{background:rgba(0,230,118,0.14);border:1px solid rgba(0,230,118,0.3);color:#00e676}
.cab-pill.warn{background:rgba(255,171,0,0.14);border:1px solid rgba(255,171,0,0.3);color:#ffca4f}
.cab-pill.danger{background:rgba(255,61,61,0.14);border:1px solid rgba(255,61,61,0.35);color:#ff6b6b}
.cab-hint{font-size:0.74rem;color:#9aa1b1;margin-top:0.45rem}
.quick-row{display:flex;gap:10px;margin-top:0.7rem}
.quick-card{flex:1;background:#11141d;border:2px solid #242a39;border-radius:12px;padding:0.8rem 0.85rem;min-height:92px;display:flex;flex-direction:column;justify-content:center}
.quick-label{font-size:0.6rem;color:#6a7389;text-transform:uppercase;letter-spacing:1.3px;font-weight:600}
.quick-val{font-size:1.15rem;color:#edf1fa;font-family:'IBM Plex Mono',monospace;font-weight:700;margin-top:0.15rem}
.quick-sub{font-size:0.7rem;color:#8b94a8;margin-top:0.15rem}
section[data-testid="stSidebar"] .stButton button{min-height:54px;font-size:1rem;font-weight:700}
section[data-testid="stSidebar"] .stToggle label p{font-size:0.9rem}
div[data-testid="stSidebar"] .stToggle{margin-bottom:0.15rem}
button[kind="secondary"],button[kind="primary"]{border-radius:10px}
#agri-scout-cam{border-radius:10px;overflow:hidden}
[data-testid="stElementContainer"]:has(#agri-scout-cam) button{min-height:48px}
[data-testid="stElementContainer"]:has(#agri-scout-cam) select{min-height:48px}
[data-testid="stElementContainer"]:has(#agri-scout-cam){background:#0c0e14;border:1px solid #1a1d28;border-radius:12px;padding:0.6rem}
@media (max-width: 1100px){
  .quick-row{flex-direction:column}
  .map-stats{flex-wrap:wrap;gap:8px}
}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
[data-testid="stMetric"]{display:none}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ==============================================================================
#  Singleton Engine
# ==============================================================================

@st.cache_resource(show_spinner="Loading Agri-Scout model…")
def get_engine():
    return AgriScoutEngine()

engine = get_engine()


# ==============================================================================
#  Session State  (plain assignments, no type annotations)
# ==============================================================================

if "last_class" not in st.session_state:
    st.session_state.last_class = "Waiting…"
if "last_conf" not in st.session_state:
    st.session_state.last_conf = 0.0
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "advisory_text" not in st.session_state:
    st.session_state.advisory_text = ""
if "disease_logs" not in st.session_state:
    st.session_state.disease_logs = []
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = 0.0
if "live_map_refresh" not in st.session_state:
    st.session_state.live_map_refresh = False
if "map_interactive" not in st.session_state:
    st.session_state.map_interactive = False
if "map_zoom_level" not in st.session_state:
    st.session_state.map_zoom_level = 17
if "map_base_type" not in st.session_state:
    st.session_state.map_base_type = "roadmap"
if st.session_state.map_base_type not in _MAP_TYPES:
    st.session_state.map_base_type = "roadmap"
if "tractor_sim" not in st.session_state:
    if _MAP_AVAILABLE:
        st.session_state.tractor_sim = TractorSimulator()
    else:
        st.session_state.tractor_sim = None
if "tractor_mode" not in st.session_state:
    st.session_state.tractor_mode = True
if "show_advanced_ui" not in st.session_state:
    st.session_state.show_advanced_ui = False
if "sunlight_mode" not in st.session_state:
    st.session_state.sunlight_mode = True
if "color_scheme" not in st.session_state:
    st.session_state.color_scheme = _COLOR_SCHEMES[0]
if "high_contrast_mode" not in st.session_state:
    st.session_state.high_contrast_mode = False
if "invert_colors_mode" not in st.session_state:
    st.session_state.invert_colors_mode = False
if "farm_name" not in st.session_state:
    st.session_state.farm_name = _DEFAULT_FARM_NAME
if "farm_lat" not in st.session_state:
    st.session_state.farm_lat = float(_DEFAULT_FARM_LAT)
if "farm_lng" not in st.session_state:
    st.session_state.farm_lng = float(_DEFAULT_FARM_LNG)
if "farm_width_m" not in st.session_state:
    st.session_state.farm_width_m = float(_DEFAULT_FARM_WIDTH_M)
if "farm_height_m" not in st.session_state:
    st.session_state.farm_height_m = float(_DEFAULT_FARM_HEIGHT_M)
if "active_farm_name" not in st.session_state:
    st.session_state.active_farm_name = st.session_state.farm_name
if "active_farm_lat" not in st.session_state:
    st.session_state.active_farm_lat = float(st.session_state.farm_lat)
if "active_farm_lng" not in st.session_state:
    st.session_state.active_farm_lng = float(st.session_state.farm_lng)
if "active_farm_width_m" not in st.session_state:
    st.session_state.active_farm_width_m = float(st.session_state.farm_width_m)
if "active_farm_height_m" not in st.session_state:
    st.session_state.active_farm_height_m = float(st.session_state.farm_height_m)
if "disease_spot_size" not in st.session_state:
    st.session_state.disease_spot_size = float(_DEFAULT_DISEASE_SPOT_SIZE)
st.session_state.disease_spot_size = float(max(3.0, min(24.0, float(st.session_state.disease_spot_size))))
if "settings_spot_size" not in st.session_state:
    st.session_state.settings_spot_size = int(round(st.session_state.disease_spot_size))
else:
    st.session_state.settings_spot_size = int(max(3, min(24, int(st.session_state.settings_spot_size))))
    st.session_state.disease_spot_size = float(st.session_state.settings_spot_size)
if "dv_marker_spot_size" not in st.session_state:
    st.session_state.dv_marker_spot_size = int(round(st.session_state.disease_spot_size))
if "map_profiles" not in st.session_state:
    st.session_state.map_profiles = [
        {
            "name": _DEFAULT_FARM_NAME,
            "farm_name": _DEFAULT_FARM_NAME,
            "lat": float(_DEFAULT_FARM_LAT),
            "lng": float(_DEFAULT_FARM_LNG),
            "width_m": float(_DEFAULT_FARM_WIDTH_M),
            "height_m": float(_DEFAULT_FARM_HEIGHT_M),
            "map_type": st.session_state.map_base_type,
            "spot_size": float(_DEFAULT_DISEASE_SPOT_SIZE),
        }
    ]
for profile in st.session_state.map_profiles:
    raw_map_type = str(profile.get("map_type", "roadmap")).lower().strip()
    profile["map_type"] = raw_map_type if raw_map_type in _MAP_TYPES else "roadmap"
    try:
        profile["spot_size"] = float(max(3.0, min(24.0, float(profile.get("spot_size", _DEFAULT_DISEASE_SPOT_SIZE)))))
    except Exception:
        profile["spot_size"] = float(_DEFAULT_DISEASE_SPOT_SIZE)
if "selected_profile_name" not in st.session_state:
    st.session_state.selected_profile_name = _DEFAULT_FARM_NAME
if "profile_name_input" not in st.session_state:
    st.session_state.profile_name_input = _DEFAULT_FARM_NAME
if "profile_map_type" not in st.session_state:
    st.session_state.profile_map_type = st.session_state.map_base_type
if st.session_state.profile_map_type not in _MAP_TYPES:
    st.session_state.profile_map_type = "roadmap"
if "settings_notice" not in st.session_state:
    st.session_state.settings_notice = ""
if "demo_video_url" not in st.session_state:
    st.session_state.demo_video_url = ""
if "gallery_limit" not in st.session_state:
    st.session_state.gallery_limit = 24
if "pending_display_preset" not in st.session_state:
    st.session_state.pending_display_preset = ""
if "pending_reset_farm" not in st.session_state:
    st.session_state.pending_reset_farm = False
if "pending_load_profile_name" not in st.session_state:
    st.session_state.pending_load_profile_name = ""
if "pending_delete_profile_name" not in st.session_state:
    st.session_state.pending_delete_profile_name = ""
if "pending_drive_coord_apply" not in st.session_state:
    st.session_state.pending_drive_coord_apply = False
if "drive_lat_input" not in st.session_state:
    st.session_state.drive_lat_input = float(st.session_state.active_farm_lat)
if "drive_lng_input" not in st.session_state:
    st.session_state.drive_lng_input = float(st.session_state.active_farm_lng)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  THREADING / INFERENCE — 100% UNCHANGED FROM v2                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

_frame_lock = threading.Lock()
_frame_slot: Optional[np.ndarray] = None
_frame_event = threading.Event()

_result_lock = threading.Lock()
_result_label: str = "Initialising…"
_result_conf: float = 0.0
_result_latency_ms: float = 0.0
_inference_count: int = 0

_INFERENCE_COOLDOWN = 0.15
_worker_running = False


def _inference_worker():
    global _result_label, _result_conf, _result_latency_ms, _inference_count
    global _frame_slot

    while _worker_running:
        triggered = _frame_event.wait(timeout=0.5)
        if not _worker_running:
            break
        if not triggered:
            continue
        _frame_event.clear()

        with _frame_lock:
            frame = _frame_slot
            _frame_slot = None

        if frame is None:
            continue

        t0 = time.perf_counter()
        label, conf = engine.process_frame(frame)
        latency = (time.perf_counter() - t0) * 1000.0

        with _result_lock:
            _result_label = label
            _result_conf = conf
            _result_latency_ms = latency
            _inference_count += 1

        time.sleep(_INFERENCE_COOLDOWN)


def _ensure_worker_started():
    global _worker_running
    if _worker_running:
        return
    _worker_running = True
    t = threading.Thread(target=_inference_worker, daemon=True, name="agri-scout-infer")
    t.start()

_ensure_worker_started()


# ==============================================================================
#  OSD Drawing — UNCHANGED
# ==============================================================================

def draw_osd(frame, label, confidence, latency_ms, infer_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    is_healthy = "healthy" in label.lower()
    accent = (64, 200, 100) if is_healthy else (60, 60, 220)
    banner_bg = (30, 60, 35) if is_healthy else (35, 25, 70)

    banner_h = 78
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_bg, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, label, (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(frame, conf_text, (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, accent, 2, cv2.LINE_AA)

    perf_text = f"Infer: {latency_ms:.0f}ms  |  #{infer_count}"
    cv2.putText(frame, perf_text, (16, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = w - 220, 18, 200, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
    cv2.putText(frame, conf_text, (bar_x + bar_w + 6, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "ASYNC", (bar_x, bar_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 130), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (2, 2), (w - 3, h - 3), accent, 2)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (w - 260, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(frame, "AGRI-SCOUT  v4", (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 160, 120), 1, cv2.LINE_AA)

    return frame


# ==============================================================================
#  WebRTC Callback — UNCHANGED
# ==============================================================================

_FRAME_SKIP = 5
_callback_counter: int = 0


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global _frame_slot, _callback_counter

    img = frame.to_ndarray(format="bgr24")

    _callback_counter += 1
    if _callback_counter % _FRAME_SKIP == 0:
        with _frame_lock:
            _frame_slot = img.copy()
        _frame_event.set()

    with _result_lock:
        label = _result_label
        conf = _result_conf
        latency = _result_latency_ms
        count = _inference_count

    st.session_state.last_class = label
    st.session_state.last_conf = conf
    st.session_state.frame_count = count

    img = draw_osd(img, label, conf, latency, count)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  GHOST TRACTOR — Position + Disease Logging                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _update_tractor_and_logs(simulator):
    """Advance tractor simulation and append disease logs with debounce."""
    tractor_lat, tractor_lng, tractor_heading = simulator.get_current_position()

    _CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    heading_cardinal = _CARDINALS[int(((tractor_heading + 22.5) % 360) / 45)]

    det_label = st.session_state.last_class
    det_conf = st.session_state.last_conf
    is_disease_detected = (
        "healthy" not in det_label.lower()
        and "waiting" not in det_label.lower()
        and "initial" not in det_label.lower()
        and det_conf >= _DISEASE_CONFIDENCE_THRESH
    )

    if is_disease_detected:
        now = time.time()
        time_ok = (now - st.session_state.last_log_time) >= _LOG_COOLDOWN_SEC

        dist_ok = True
        logs = st.session_state.disease_logs
        if logs:
            last = logs[-1]
            dist_ok = _haversine_m(last.lat, last.lng, tractor_lat, tractor_lng) >= _LOG_MIN_DISTANCE_M

        if time_ok and dist_ok:
            st.session_state.disease_logs.append(DiseaseLog(
                lat=tractor_lat,
                lng=tractor_lng,
                label=det_label,
                confidence=det_conf,
                timestamp=now,
            ))
            st.session_state.last_log_time = now

    disease_count = len(st.session_state.disease_logs)
    return tractor_lat, tractor_lng, tractor_heading, heading_cardinal, disease_count


# Default values (used if map_engine is unavailable)
tractor_lat = 43.1580
tractor_lng = -89.9120
tractor_heading = 90.0
heading_cardinal = "E"
disease_count = 0

sim = st.session_state.tractor_sim

if sim is not None:
    tractor_lat, tractor_lng, tractor_heading, heading_cardinal, disease_count = _update_tractor_and_logs(sim)
else:
    disease_count = len(st.session_state.disease_logs)

# Pre-build HTML snippets (avoids escaped-quote issues in f-strings)
lat_str = f"{abs(tractor_lat):.5f}"
lng_str = f"{abs(tractor_lng):.5f}"
lat_dir = "N" if tractor_lat >= 0 else "S"
lng_dir = "W" if tractor_lng < 0 else "E"
red_dot = '<span class="live-dot red"></span>' if disease_count > 0 else ""
farm_name_display = (st.session_state.active_farm_name or _DEFAULT_FARM_NAME).strip()
if not farm_name_display:
    farm_name_display = _DEFAULT_FARM_NAME
farm_name_display = (
    farm_name_display
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)
last_det_age_text = "None yet"
last_det_dist_text = "--"

if st.session_state.disease_logs:
    _last_det = st.session_state.disease_logs[-1]
    _age_sec = max(0, int(time.time() - _last_det.timestamp))
    if _age_sec < 60:
        last_det_age_text = f"{_age_sec}s ago"
    elif _age_sec < 3600:
        last_det_age_text = f"{_age_sec // 60}m ago"
    else:
        last_det_age_text = f"{_age_sec // 3600}h ago"

    _dist_m = _haversine_m(tractor_lat, tractor_lng, _last_det.lat, _last_det.lng)
    if _dist_m < 1000:
        last_det_dist_text = f"{_dist_m:.0f} m"
    else:
        last_det_dist_text = f"{_dist_m / 1000:.2f} km"


def _build_theme_override_css():
    scheme = st.session_state.color_scheme

    palettes = {
        "Night Field (Dark)": {
            "app_bg": "#08090c",
            "panel": "#10121a",
            "panel_alt": "#0c0e14",
            "sidebar": "#0c0e14",
            "text": "#eaedf3",
            "muted": "#8f95a5",
            "border": "#1f2636",
            "accent": "#00e676",
            "danger": "#ff6b6b",
        },
        "Daylight Paper (Black on Light)": {
            "app_bg": "#fffdf4",
            "panel": "#ffffff",
            "panel_alt": "#fff9df",
            "sidebar": "#f6f0d4",
            "text": "#101010",
            "muted": "#353535",
            "border": "#b5ab84",
            "accent": "#0b7a50",
            "danger": "#9d1f1f",
        },
        "Green Cabin (White on Dark Green)": {
            "app_bg": "#07110d",
            "panel": "#10221a",
            "panel_alt": "#0e1a26",
            "sidebar": "#0a1812",
            "text": "#f0fff6",
            "muted": "#b5cebf",
            "border": "#2d5846",
            "accent": "#8ef8c4",
            "danger": "#ff9d9d",
        },
        "Amber Contrast (Black on Amber)": {
            "app_bg": "#2a1e00",
            "panel": "#ffd85a",
            "panel_alt": "#ffca28",
            "sidebar": "#ffcc3d",
            "text": "#0f0f0f",
            "muted": "#272727",
            "border": "#8d6800",
            "accent": "#0f0f0f",
            "danger": "#4d0000",
        },
    }

    def _hex_to_rgb(hex_color):
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _luma(rgb):
        r, g, b = rgb
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    p = palettes.get(scheme, palettes["Night Field (Dark)"])
    if st.session_state.high_contrast_mode:
        panel_luma = _luma(_hex_to_rgb(p["panel"]))
        contrast_text = "#000000" if panel_luma >= 0.5 else "#ffffff"
        contrast_muted = "#1a1a1a" if panel_luma >= 0.5 else "#f2f5ff"
    else:
        contrast_text = p["text"]
        contrast_muted = p["muted"]
    border_width = "2px" if st.session_state.high_contrast_mode else "1px"

    invert_css = ""
    if st.session_state.invert_colors_mode:
        invert_css = ".stApp{filter:invert(1) hue-rotate(180deg) !important;}"

    return f"""
    <style>
    :root {{
      --ag-app-bg: {p["app_bg"]};
      --ag-panel: {p["panel"]};
      --ag-panel-alt: {p["panel_alt"]};
      --ag-sidebar: {p["sidebar"]};
      --ag-text: {contrast_text};
      --ag-muted: {contrast_muted};
      --ag-border: {p["border"]};
      --ag-accent: {p["accent"]};
      --ag-danger: {p["danger"]};
    }}
    .stApp{{background:var(--ag-app-bg) !important;color:var(--ag-text) !important;}}
    section[data-testid="stSidebar"]{{background:var(--ag-sidebar) !important;border-right:{border_width} solid var(--ag-border) !important;}}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{{color:var(--ag-muted) !important;}}
    .hdr,.tcard,.det-panel.waiting,.advisory-panel,.adv-body,.map-panel,.mcard,.quick-card,[data-testid="stElementContainer"]:has(#agri-scout-cam){{background:var(--ag-panel) !important;border:{border_width} solid var(--ag-border) !important;}}
    .feed-wrap{{background:var(--ag-panel-alt) !important;border:{border_width} solid var(--ag-border) !important;}}
    .hdr h1,.det-class,.tc-val,.quick-val,.adv-content,.adv-title,.map-title,.cab-alert-state,.cab-alert-meta{{color:var(--ag-text) !important;}}
    .tc-label,.tc-sub,.quick-label,.quick-sub,.feed-label,.adv-status,.adv-footer-item,.det-meta,.map-stat{{color:var(--ag-muted) !important;}}
    .tc-unit,.chip-live,.live-dot.green{{color:var(--ag-accent) !important;}}
    .chip-live,.chip-npu,.cab-pill.safe,.cab-pill.warn,.cab-pill.danger{{border:{border_width} solid var(--ag-border) !important;}}
    .cab-alert.warn,.cab-alert.safe,.cab-alert.danger{{border:{border_width} solid var(--ag-border) !important;}}
    .cab-pill.safe{{background:rgba(0,230,118,0.2) !important;}}
    .cab-pill.warn{{background:rgba(255,171,0,0.2) !important;color:var(--ag-text) !important;}}
    .cab-pill.danger{{background:rgba(255,61,61,0.2) !important;color:var(--ag-danger) !important;}}
    .stButton button{{border:{border_width} solid var(--ag-border) !important;}}
    {invert_css}
    </style>
    """


def _apply_display_preset(preset_name):
    if preset_name == "Day":
        st.session_state.color_scheme = "Daylight Paper (Black on Light)"
        st.session_state.high_contrast_mode = False
        st.session_state.invert_colors_mode = False
        st.session_state.sunlight_mode = True
    elif preset_name == "Emergency":
        st.session_state.color_scheme = "Amber Contrast (Black on Amber)"
        st.session_state.high_contrast_mode = True
        st.session_state.invert_colors_mode = False
        st.session_state.sunlight_mode = True
    else:
        st.session_state.color_scheme = "Night Field (Dark)"
        st.session_state.high_contrast_mode = False
        st.session_state.invert_colors_mode = False
        st.session_state.sunlight_mode = False


def _coerce_map_type(raw_map_type):
    map_type = str(raw_map_type).lower().strip()
    return map_type if map_type in _MAP_TYPES else "roadmap"


def _apply_farm_location():
    if not _MAP_AVAILABLE or TractorSimulator is None:
        return False, "Map simulator is unavailable in this session."

    try:
        lat = float(st.session_state.farm_lat)
        lng = float(st.session_state.farm_lng)
        field_w_m = float(st.session_state.farm_width_m)
        field_h_m = float(st.session_state.farm_height_m)
    except Exception:
        return False, "Coordinates and farm size must be numeric values."

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return False, "Coordinates out of range. Use latitude [-90, 90] and longitude [-180, 180]."
    if field_w_m < 40.0 or field_h_m < 40.0:
        return False, "Farm width and height must be at least 40 meters."
    if field_w_m > 20_000.0 or field_h_m > 20_000.0:
        return False, "Farm width/height too large for this simulator. Keep each under 20,000 meters."

    # Guard against map math degenerating near poles or world edges.
    if abs(lat) > 84.0:
        return False, "Latitude too close to poles for stable map rendering. Use a value between -84 and 84."

    m_per_deg_lat = 111_320.0
    m_per_deg_lng = m_per_deg_lat * math.cos(math.radians(lat))
    if m_per_deg_lng < 1_000.0:
        return False, "Longitude scale is unstable at this latitude. Choose a latitude further from the poles."

    south_lat = lat - (field_h_m / m_per_deg_lat)
    east_lng = lng + (field_w_m / m_per_deg_lng)
    if south_lat < -85.0 or south_lat > 85.0:
        return False, "Location places the simulated field outside map bounds. Adjust latitude."
    if east_lng < -179.95 or east_lng > 179.95:
        return False, "Location places the simulated field outside longitude bounds. Adjust longitude."

    current_speed = 5.0
    if st.session_state.tractor_sim is not None:
        current_speed = st.session_state.tractor_sim.get_speed_mph()

    st.session_state.tractor_sim = TractorSimulator(
        start_lat=lat,
        start_lng=lng,
        speed_mph=current_speed,
        field_width_m=field_w_m,
        field_height_m=field_h_m,
    )
    st.session_state.active_farm_name = (st.session_state.farm_name or _DEFAULT_FARM_NAME).strip() or _DEFAULT_FARM_NAME
    st.session_state.active_farm_lat = lat
    st.session_state.active_farm_lng = lng
    st.session_state.active_farm_width_m = field_w_m
    st.session_state.active_farm_height_m = field_h_m
    st.session_state.drive_lat_input = lat
    st.session_state.drive_lng_input = lng
    st.session_state.disease_logs = []
    st.session_state.last_log_time = 0.0
    return True, "Farm location updated. Map and tractor path reset."


def _save_current_profile(profile_name):
    name = (profile_name or "").strip()
    if not name:
        return False, "Profile name cannot be empty."

    profile_map_type = _coerce_map_type(st.session_state.get("profile_map_type", st.session_state.map_base_type))
    profile = {
        "name": name,
        "farm_name": (st.session_state.farm_name or _DEFAULT_FARM_NAME).strip() or _DEFAULT_FARM_NAME,
        "lat": float(st.session_state.farm_lat),
        "lng": float(st.session_state.farm_lng),
        "width_m": float(st.session_state.farm_width_m),
        "height_m": float(st.session_state.farm_height_m),
        "map_type": profile_map_type,
        "spot_size": float(st.session_state.disease_spot_size),
    }

    replaced = False
    for i, existing in enumerate(st.session_state.map_profiles):
        if existing.get("name") == name:
            st.session_state.map_profiles[i] = profile
            replaced = True
            break
    if not replaced:
        st.session_state.map_profiles.append(profile)
    st.session_state.selected_profile_name = name
    return True, "Profile saved."


def _load_profile_by_name(profile_name):
    for p in st.session_state.map_profiles:
        if p.get("name") == profile_name:
            loaded_map_type = _coerce_map_type(p.get("map_type", "roadmap"))
            loaded_spot_size = float(max(3.0, min(24.0, float(p.get("spot_size", _DEFAULT_DISEASE_SPOT_SIZE)))))
            st.session_state.selected_profile_name = profile_name
            st.session_state.profile_name_input = profile_name
            st.session_state.farm_name = p.get("farm_name", _DEFAULT_FARM_NAME)
            st.session_state.farm_lat = float(p.get("lat", _DEFAULT_FARM_LAT))
            st.session_state.farm_lng = float(p.get("lng", _DEFAULT_FARM_LNG))
            st.session_state.farm_width_m = float(p.get("width_m", _DEFAULT_FARM_WIDTH_M))
            st.session_state.farm_height_m = float(p.get("height_m", _DEFAULT_FARM_HEIGHT_M))
            st.session_state.map_base_type = loaded_map_type
            st.session_state.profile_map_type = loaded_map_type
            st.session_state.disease_spot_size = loaded_spot_size
            st.session_state.settings_spot_size = int(round(loaded_spot_size))
            st.session_state.dv_marker_spot_size = int(round(loaded_spot_size))
            return _apply_farm_location()
    return False, f"Profile '{profile_name}' was not found."


def _delete_profile_by_name(profile_name):
    profiles = st.session_state.map_profiles
    if len(profiles) <= 1:
        return False, "At least one profile must remain."
    new_profiles = [p for p in profiles if p.get("name") != profile_name]
    if len(new_profiles) == len(profiles):
        return False, f"Profile '{profile_name}' was not found."
    st.session_state.map_profiles = new_profiles
    first = new_profiles[0]
    st.session_state.selected_profile_name = first.get("name", _DEFAULT_FARM_NAME)
    st.session_state.profile_name_input = st.session_state.selected_profile_name
    st.session_state.profile_map_type = _coerce_map_type(first.get("map_type", st.session_state.map_base_type))
    return True, f"Deleted profile '{profile_name}'."


def _apply_pending_actions():
    preset = st.session_state.get("pending_display_preset", "")
    if preset:
        _apply_display_preset(preset)
        st.session_state.pending_display_preset = ""

    if st.session_state.get("pending_reset_farm", False):
        st.session_state.farm_name = _DEFAULT_FARM_NAME
        st.session_state.farm_lat = float(_DEFAULT_FARM_LAT)
        st.session_state.farm_lng = float(_DEFAULT_FARM_LNG)
        st.session_state.farm_width_m = float(_DEFAULT_FARM_WIDTH_M)
        st.session_state.farm_height_m = float(_DEFAULT_FARM_HEIGHT_M)
        st.session_state.map_base_type = "roadmap"
        st.session_state.profile_map_type = "roadmap"
        st.session_state.disease_spot_size = float(_DEFAULT_DISEASE_SPOT_SIZE)
        st.session_state.settings_spot_size = int(round(_DEFAULT_DISEASE_SPOT_SIZE))
        st.session_state.dv_marker_spot_size = int(round(_DEFAULT_DISEASE_SPOT_SIZE))
        ok, msg = _apply_farm_location()
        st.session_state.settings_notice = msg if ok else "Default reset attempted, but simulator is unavailable."
        st.session_state.pending_reset_farm = False

    load_name = st.session_state.get("pending_load_profile_name", "")
    if load_name:
        ok, msg = _load_profile_by_name(load_name)
        st.session_state.settings_notice = msg
        st.session_state.pending_load_profile_name = ""

    delete_name = st.session_state.get("pending_delete_profile_name", "")
    if delete_name:
        ok, msg = _delete_profile_by_name(delete_name)
        st.session_state.settings_notice = msg
        st.session_state.pending_delete_profile_name = ""

    if st.session_state.get("pending_drive_coord_apply", False):
        st.session_state.farm_lat = float(st.session_state.drive_lat_input)
        st.session_state.farm_lng = float(st.session_state.drive_lng_input)
        ok, msg = _apply_farm_location()
        st.session_state.settings_notice = msg
        st.session_state.pending_drive_coord_apply = False


@st.cache_data(show_spinner=False, ttl=120)
def _scan_gallery_media(limit=200):
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    roots = [Path("recordings"), Path("captures"), Path("dataset")]
    items = []

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in image_exts and ext not in video_exts:
                continue
            kind = "video" if ext in video_exts else "image"
            items.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "kind": kind,
                    "ext": ext,
                    "size_kb": round(p.stat().st_size / 1024.0, 1),
                    "mtime": p.stat().st_mtime,
                    "parent": p.parent.name,
                }
            )
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


@st.cache_data(show_spinner=False, ttl=180)
def _build_sample_farm_dataset(center_lat, center_lng, farm_width_m, farm_height_m, n_points=260, seed=42):
    rng = random.Random(f"{center_lat:.5f}:{center_lng:.5f}:{farm_width_m:.1f}:{farm_height_m:.1f}:{n_points}:{seed}")
    diseases = [
        "Corn Rust",
        "Northern Leaf Blight",
        "Gray Leaf Spot",
        "Anthracnose",
    ]

    m_per_deg_lat = 111_320.0
    m_per_deg_lng = max(1_000.0, m_per_deg_lat * math.cos(math.radians(center_lat)))
    lat_span = max(0.00025, float(farm_height_m) / m_per_deg_lat)
    lng_span = max(0.00025, float(farm_width_m) / m_per_deg_lng)
    cluster_lat_span = lat_span * 0.45
    cluster_lng_span = lng_span * 0.45

    clusters = []
    for _ in range(5):
        clusters.append(
            {
                "lat_off": rng.uniform(-cluster_lat_span, cluster_lat_span),
                "lng_off": rng.uniform(-cluster_lng_span, cluster_lng_span),
                "pressure": rng.uniform(0.45, 0.95),
            }
        )

    rows = []
    now = datetime.datetime.now()
    for i in range(max(50, int(n_points))):
        c = rng.choice(clusters)
        lat = center_lat + rng.gauss(c["lat_off"], max(0.00005, lat_span * 0.12))
        lng = center_lng + rng.gauss(c["lng_off"], max(0.00005, lng_span * 0.12))

        moisture_pct = max(22.0, min(99.0, rng.gauss(67.0, 11.0)))
        humidity_pct = max(30.0, min(100.0, rng.gauss(74.0, 10.0)))
        temp_c = max(10.0, min(43.0, rng.gauss(27.0, 4.2)))
        crop_stress_pct = max(5.0, min(100.0, rng.gauss(41.0, 18.0)))

        raw_alert = (
            c["pressure"] * 100.0
            + (moisture_pct - 60.0) * 0.24
            + (humidity_pct - 70.0) * 0.30
            + (crop_stress_pct - 40.0) * 0.18
            + rng.gauss(0.0, 8.5)
        )
        disease_alert_pct = max(3.0, min(99.0, raw_alert))
        disease_label = rng.choices(diseases, weights=[0.34, 0.30, 0.24, 0.12], k=1)[0]
        risk_band = (
            "High" if disease_alert_pct >= 70
            else "Moderate" if disease_alert_pct >= 40
            else "Low"
        )
        ts = now - datetime.timedelta(hours=rng.uniform(0.0, 360.0))
        zone = "Z" + str(1 + (i % 8))

        rows.append(
            {
                "lat": round(lat, 7),
                "lng": round(lng, 7),
                "disease_label": disease_label,
                "disease_alert_pct": round(disease_alert_pct, 1),
                "moisture_pct": round(moisture_pct, 1),
                "humidity_pct": round(humidity_pct, 1),
                "temp_c": round(temp_c, 1),
                "crop_stress_pct": round(crop_stress_pct, 1),
                "risk_band": risk_band,
                "zone": zone,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            }
        )

    rows.sort(key=lambda x: x["disease_alert_pct"], reverse=True)
    return rows


def _build_analytics_heatmap_html(
    api_key,
    center_lat,
    center_lng,
    points,
    map_height=560,
    map_type="roadmap",
    heat_opacity=0.72,
    heat_radius=30,
    marker_threshold=70,
    marker_scale=8.0,
):
    safe_map_type = str(map_type).lower().strip()
    if safe_map_type not in _MAP_TYPES:
        safe_map_type = "roadmap"

    clat = max(-85.0, min(85.0, float(center_lat)))
    clng = max(-179.9999, min(179.9999, float(center_lng)))
    h = max(320, min(900, int(map_height)))
    opacity = max(0.1, min(1.0, float(heat_opacity)))
    radius = max(10, min(70, int(heat_radius)))
    threshold = max(1.0, min(99.0, float(marker_threshold)))
    marker_size = max(2.0, min(24.0, float(marker_scale)))

    serialised = []
    for p in points:
        alert = max(0.0, min(100.0, float(p.get("disease_alert_pct", 0.0))))
        serialised.append(
            {
                "lat": max(-85.0, min(85.0, float(p.get("lat", clat)))),
                "lng": max(-179.9999, min(179.9999, float(p.get("lng", clng)))),
                "alert": alert,
                "weight": max(0.01, min(1.0, alert / 100.0)),
                "label": str(p.get("disease_label", "Disease Alert")).replace("<", "").replace(">", ""),
                "zone": str(p.get("zone", "Z?")).replace("<", "").replace(">", ""),
                "risk": str(p.get("risk_band", "Unknown")).replace("<", "").replace(">", ""),
            }
        )
    points_js = json.dumps(serialised)

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        'html,body{background:#08090c}'
        '#map{width:100%;height:' + str(h) + 'px;border-radius:10px}'
        ".hp{font-family:monospace;background:#10121a;border:1px solid #25304b;"
        "border-radius:8px;padding:8px 10px;color:#eaedf3;font-size:11px;line-height:1.45}"
        ".hp b{color:#ff7a7a}"
        ".hp i{color:#9ba3b8;font-style:normal}"
        '</style><script>'
        "const DATA=" + points_js + ";"
        "function initMap(){"
        "const center={lat:" + str(clat) + ",lng:" + str(clng) + "};"
        "const map=new google.maps.Map(document.getElementById('map'),{"
        "center:center,zoom:16,mapTypeId:'" + safe_map_type + "',disableDefaultUI:false,gestureHandling:'greedy'});"
        "const heat=[];"
        "DATA.forEach((d)=>{"
        "const pos=new google.maps.LatLng(d.lat,d.lng);"
        "heat.push({location:pos,weight:d.weight});"
        "if(d.alert>=" + str(threshold) + "){"
        "const m=new google.maps.Marker({position:pos,map:map,"
        "icon:{path:google.maps.SymbolPath.CIRCLE,scale:" + str(marker_size) + ",fillColor:'#ff3d3d',fillOpacity:0.75,strokeColor:'#ffffff',strokeWeight:1.5}});"
        "const w=new google.maps.InfoWindow({content:'<div class=\"hp\"><b>'+d.label+'</b><br><i>Alert: '+d.alert.toFixed(1)+'%</i><br><i>Zone: '+d.zone+'  •  '+d.risk+'</i></div>'});"
        "m.addListener('click',()=>w.open(map,m));"
        "}"
        "});"
        "new google.maps.visualization.HeatmapLayer({"
        "data:heat,map:map,opacity:" + str(opacity) + ",radius:" + str(radius) + ","
        "gradient:['rgba(0,0,0,0)','rgba(58,123,255,0.35)','rgba(58,123,255,0.58)','rgba(255,173,66,0.82)','rgba(255,61,61,0.98)']"
        "});"
        "}"
        "</script>"
        '<script defer src="https://maps.googleapis.com/maps/api/js?key='
        + api_key + '&libraries=visualization&callback=initMap"></script>'
        "</head><body><div id=\"map\"></div></body></html>"
    )


def _render_data_visualization_page():
    st.markdown("### Farm Disease Heatmap (Sample Dataset)")
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
    sample_size = ctrl1.slider("Sample points", 120, 1200, 320, 40, key="dv_sample_size")
    heat_opacity = ctrl2.slider("Heat opacity", 0.20, 1.00, 0.74, 0.02, key="dv_heat_opacity")
    heat_radius = ctrl3.slider("Heat radius", 12, 64, 30, 2, key="dv_heat_radius")
    marker_threshold = ctrl4.slider("Marker threshold %", 40, 95, 70, 1, key="dv_marker_threshold")
    marker_spot_size = ctrl5.slider("Spot size", 3, 22, int(st.session_state.dv_marker_spot_size), 1, key="dv_marker_spot_size")

    c5, c6 = st.columns(2)
    map_type = c5.selectbox(
        "Basemap",
        _MAP_TYPES,
        index=_MAP_TYPES.index(_coerce_map_type(st.session_state.map_base_type)),
        key="dv_map_type",
    )
    sample_seed = c6.number_input("Sample seed", min_value=1, max_value=9999, value=42, step=1, key="dv_sample_seed")

    dataset = _build_sample_farm_dataset(
        center_lat=float(st.session_state.active_farm_lat),
        center_lng=float(st.session_state.active_farm_lng),
        farm_width_m=float(st.session_state.active_farm_width_m),
        farm_height_m=float(st.session_state.active_farm_height_m),
        n_points=int(sample_size),
        seed=int(sample_seed),
    )

    heatmap_html = _build_analytics_heatmap_html(
        api_key=GOOGLE_MAPS_API_KEY,
        center_lat=float(st.session_state.active_farm_lat),
        center_lng=float(st.session_state.active_farm_lng),
        points=dataset,
        map_height=560,
        map_type=map_type,
        heat_opacity=float(heat_opacity),
        heat_radius=int(heat_radius),
        marker_threshold=float(marker_threshold),
        marker_scale=float(marker_spot_size),
    )
    components.html(heatmap_html, height=580, scrolling=False)

    alerts = [x["disease_alert_pct"] for x in dataset]
    high_alerts = [x for x in dataset if x["disease_alert_pct"] >= 70]
    avg_alert = sum(alerts) / max(1, len(alerts))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Events", len(dataset))
    c2.metric("Avg Alert", f"{avg_alert:.1f}%")
    c3.metric("High Alerts (>=70%)", len(high_alerts))
    c4.metric("Farm", st.session_state.active_farm_name)

    disease_counts = {}
    disease_alert_sum = {}
    risk_counts = {}
    zone_counts = {}
    env_totals = {"moisture_pct": 0.0, "humidity_pct": 0.0, "temp_c": 0.0, "crop_stress_pct": 0.0}
    for row in dataset:
        disease_counts[row["disease_label"]] = disease_counts.get(row["disease_label"], 0) + 1
        disease_alert_sum[row["disease_label"]] = disease_alert_sum.get(row["disease_label"], 0.0) + row["disease_alert_pct"]
        risk_counts[row["risk_band"]] = risk_counts.get(row["risk_band"], 0) + 1
        zone_counts[row["zone"]] = zone_counts.get(row["zone"], 0) + 1
        env_totals["moisture_pct"] += row["moisture_pct"]
        env_totals["humidity_pct"] += row["humidity_pct"]
        env_totals["temp_c"] += row["temp_c"]
        env_totals["crop_stress_pct"] += row["crop_stress_pct"]

    st.markdown("#### Disease Type Distribution Graphs")
    dist_mode = st.selectbox(
        "Distribution view",
        ["Bar chart", "Pie chart", "Average alert by disease"],
        key="dv_dist_mode",
    )

    dist_rows = []
    for disease in sorted(disease_counts.keys()):
        count = disease_counts[disease]
        avg_disease_alert = disease_alert_sum[disease] / max(1, count)
        dist_rows.append({"Disease": disease, "Count": count, "AvgAlert": round(avg_disease_alert, 1)})

    if dist_mode == "Pie chart":
        st.vega_lite_chart(
            {"values": dist_rows},
            {
                "mark": {"type": "arc", "innerRadius": 35},
                "encoding": {
                    "theta": {"field": "Count", "type": "quantitative"},
                    "color": {"field": "Disease", "type": "nominal"},
                    "tooltip": [{"field": "Disease"}, {"field": "Count"}],
                },
            },
            use_container_width=True,
        )
    elif dist_mode == "Average alert by disease":
        st.vega_lite_chart(
            {"values": dist_rows},
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "Disease", "type": "nominal", "sort": "-y"},
                    "y": {"field": "AvgAlert", "type": "quantitative", "title": "Avg Alert %"},
                    "color": {"field": "Disease", "type": "nominal", "legend": None},
                    "tooltip": [{"field": "Disease"}, {"field": "AvgAlert"}],
                },
            },
            use_container_width=True,
        )
    else:
        st.vega_lite_chart(
            {"values": dist_rows},
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "Disease", "type": "nominal", "sort": "-y"},
                    "y": {"field": "Count", "type": "quantitative"},
                    "color": {"field": "Disease", "type": "nominal", "legend": None},
                    "tooltip": [{"field": "Disease"}, {"field": "Count"}],
                },
            },
            use_container_width=True,
        )

    st.markdown("#### Additional Analytics Graphs")
    extra_mode = st.selectbox(
        "Additional graph",
        ["Risk bands", "Environmental averages", "Alert histogram", "Top zones by detections"],
        key="dv_extra_mode",
    )

    if extra_mode == "Risk bands":
        risk_rows = [{"Band": k, "Count": v} for k, v in sorted(risk_counts.items())]
        st.vega_lite_chart(
            {"values": risk_rows},
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "Band", "type": "nominal"},
                    "y": {"field": "Count", "type": "quantitative"},
                    "color": {"field": "Band", "type": "nominal"},
                },
            },
            use_container_width=True,
        )
    elif extra_mode == "Environmental averages":
        n = max(1, len(dataset))
        env_rows = [
            {"Metric": "Moisture %", "Value": round(env_totals["moisture_pct"] / n, 2)},
            {"Metric": "Humidity %", "Value": round(env_totals["humidity_pct"] / n, 2)},
            {"Metric": "Temperature C", "Value": round(env_totals["temp_c"] / n, 2)},
            {"Metric": "Crop Stress %", "Value": round(env_totals["crop_stress_pct"] / n, 2)},
        ]
        st.vega_lite_chart(
            {"values": env_rows},
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "Metric", "type": "nominal"},
                    "y": {"field": "Value", "type": "quantitative"},
                    "color": {"field": "Metric", "type": "nominal", "legend": None},
                },
            },
            use_container_width=True,
        )
    elif extra_mode == "Alert histogram":
        hist_rows = []
        for start in range(0, 100, 10):
            end = start + 10
            count_bin = sum(1 for a in alerts if (a >= start and (a < end or end == 100)))
            hist_rows.append({"Range": f"{start}-{end}", "Count": count_bin})
        st.vega_lite_chart(
            {"values": hist_rows},
            {
                "mark": {"type": "bar"},
                "encoding": {
                    "x": {"field": "Range", "type": "ordinal"},
                    "y": {"field": "Count", "type": "quantitative"},
                    "tooltip": [{"field": "Range"}, {"field": "Count"}],
                },
            },
            use_container_width=True,
        )
    else:
        top_zones = sorted(zone_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        zone_rows = [{"Zone": z, "Count": c} for z, c in top_zones]
        st.vega_lite_chart(
            {"values": zone_rows},
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "Zone", "type": "nominal"},
                    "y": {"field": "Count", "type": "quantitative"},
                    "color": {"field": "Zone", "type": "nominal", "legend": None},
                },
            },
            use_container_width=True,
        )

    st.markdown("#### Top Hotspots (Sample)")
    st.dataframe(
        [
            {
                "Timestamp": x["timestamp"],
                "Disease": x["disease_label"],
                "Alert %": x["disease_alert_pct"],
                "Risk": x["risk_band"],
                "Zone": x["zone"],
                "Moisture %": x["moisture_pct"],
                "Humidity %": x["humidity_pct"],
                "Temp C": x["temp_c"],
                "Lat": x["lat"],
                "Lng": x["lng"],
            }
            for x in dataset[:35]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_compute_page():
    st.markdown("### Power and Efficiency Overview")
    st.caption("Current values are dashboard-estimated placeholders until hardware telemetry integration is added.")
    cpu_pct = min(95, 25 + (st.session_state.frame_count % 35))
    gpu_pct = min(95, 18 + (st.session_state.frame_count % 28))
    npu_pct = min(95, 35 + (st.session_state.frame_count % 42))
    c1, c2, c3 = st.columns(3)
    c1.metric("CPU Utilization", f"{cpu_pct}%")
    c2.metric("GPU Utilization", f"{gpu_pct}%")
    c3.metric("NPU Utilization", f"{npu_pct}%")
    st.progress(npu_pct / 100.0, text=f"NPU load {npu_pct}%")
    st.progress(gpu_pct / 100.0, text=f"GPU load {gpu_pct}%")
    st.progress(cpu_pct / 100.0, text=f"CPU load {cpu_pct}%")
    efficiency = round((npu_pct / max(1, cpu_pct + gpu_pct)) * 100, 1)
    st.metric("NPU Efficiency Ratio", f"{efficiency}%")


def _render_gallery_page():
    st.markdown("### Recordings and Images")
    f1, f2, f3 = st.columns([1, 1, 1])
    media_type = f1.selectbox("Type", ["all", "image", "video"], index=0, key="gallery_type_filter")
    source = f2.selectbox("Source", ["all", "recordings/captures", "dataset"], index=0, key="gallery_source_filter")
    st.session_state.gallery_limit = int(
        f3.slider("Max items", min_value=12, max_value=96, value=int(st.session_state.gallery_limit), step=12, key="gallery_limit_slider")
    )

    items = _scan_gallery_media(limit=max(120, int(st.session_state.gallery_limit)))
    if source != "all":
        if source == "dataset":
            items = [x for x in items if x["path"].startswith("dataset" + os.sep) or x["path"].startswith("Agri-Scout" + os.sep + "dataset")]
        else:
            items = [x for x in items if x["path"].startswith("recordings" + os.sep) or x["path"].startswith("captures" + os.sep)]
    if media_type != "all":
        items = [x for x in items if x["kind"] == media_type]

    items = items[: int(st.session_state.gallery_limit)]
    st.caption(f"Showing {len(items)} items")
    if not items:
        st.info("No media matched your filters.")
    else:
        for idx, item in enumerate(items):
            st.markdown(f"**{item['name']}**  •  `{item['kind']}`  •  {item['size_kb']} KB  •  `{item['parent']}`")
            try:
                if item["kind"] == "image":
                    st.image(item["path"], use_container_width=True)
                else:
                    st.video(item["path"])
            except Exception as e:
                st.warning(f"Could not render `{item['path']}`: {e}")
            if idx < len(items) - 1:
                st.markdown("---")


def _render_settings_page():
    st.markdown("### Settings")
    if st.session_state.settings_notice:
        st.success(st.session_state.settings_notice)
        st.session_state.settings_notice = ""

    st.markdown("#### Display Presets")
    p1, p2, p3 = st.columns(3)
    if p1.button("🌙 Night", use_container_width=True, key="tab_preset_night"):
        st.session_state.pending_display_preset = "Night"
        st.rerun()
    if p2.button("☀️ Day", use_container_width=True, key="tab_preset_day"):
        st.session_state.pending_display_preset = "Day"
        st.rerun()
    if p3.button("🚨 Emergency", use_container_width=True, key="tab_preset_emergency"):
        st.session_state.pending_display_preset = "Emergency"
        st.rerun()

    scheme_index = _COLOR_SCHEMES.index(st.session_state.color_scheme) if st.session_state.color_scheme in _COLOR_SCHEMES else 0
    st.selectbox(
        "Color scheme",
        options=_COLOR_SCHEMES,
        index=scheme_index,
        key="color_scheme",
    )
    st.toggle(
        "High contrast mode",
        value=st.session_state.high_contrast_mode,
        key="high_contrast_mode",
    )
    st.toggle(
        "Invert colors",
        value=st.session_state.invert_colors_mode,
        key="invert_colors_mode",
    )
    if st.session_state.invert_colors_mode:
        st.caption("Invert mode affects the entire screen, including camera and map colors.")
    st.caption("Tip: set device brightness to 100% for strongest daylight readability.")

    st.markdown("#### Farm Location")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.text_input("Farm name", key="farm_name")
    r1c2.number_input("Latitude", key="farm_lat", min_value=-90.0, max_value=90.0, step=0.0001, format="%.6f")
    r1c3.number_input("Longitude", key="farm_lng", min_value=-180.0, max_value=180.0, step=0.0001, format="%.6f")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.number_input("Farm width (m)", key="farm_width_m", min_value=40.0, max_value=20000.0, step=10.0, format="%.1f")
    r2c2.number_input("Farm height (m)", key="farm_height_m", min_value=40.0, max_value=20000.0, step=10.0, format="%.1f")
    r2c3.slider("Disease spot size", min_value=3, max_value=24, value=int(st.session_state.settings_spot_size), step=1, key="settings_spot_size")

    area_acres = (float(st.session_state.farm_width_m) * float(st.session_state.farm_height_m)) / 4046.8564224
    st.caption(f"Estimated field area: {area_acres:.2f} acres")

    a1, a2 = st.columns(2)
    if a1.button("Apply Farm Location", use_container_width=True, key="tab_apply_farm_location"):
        ok, msg = _apply_farm_location()
        if ok:
            st.session_state.settings_notice = msg
            st.rerun()
        else:
            st.warning(msg)
    if a2.button("Reset To Default Farm", use_container_width=True, key="tab_reset_farm_location"):
        st.session_state.pending_reset_farm = True
        st.rerun()

    st.markdown("#### Saved Map Profiles")
    p1, p2, p3 = st.columns([1.5, 1.2, 1.1])
    p1.text_input("Profile name", key="profile_name_input", placeholder="e.g. North-40 Hybrid")
    p2.selectbox(
        "Profile map layer",
        options=_MAP_TYPES,
        index=_MAP_TYPES.index(_coerce_map_type(st.session_state.profile_map_type)),
        key="profile_map_type",
    )
    if p3.button("Save / Update Profile", use_container_width=True, key="tab_save_profile"):
        ok, msg = _save_current_profile(st.session_state.profile_name_input)
        st.session_state.settings_notice = msg
        st.rerun()

    profile_names = [p.get("name", "Unnamed") for p in st.session_state.map_profiles]
    if st.session_state.selected_profile_name not in profile_names and profile_names:
        st.session_state.selected_profile_name = profile_names[0]
    st.selectbox("Select saved profile", options=profile_names, key="selected_profile_name")

    p3, p4 = st.columns(2)
    if p3.button("Load Selected Profile", use_container_width=True, key="tab_load_profile"):
        st.session_state.pending_load_profile_name = st.session_state.selected_profile_name
        st.rerun()
    if p4.button("Delete Selected Profile", use_container_width=True, key="tab_delete_profile"):
        st.session_state.pending_delete_profile_name = st.session_state.selected_profile_name
        st.rerun()

    st.dataframe(
        [
            {
                "Profile": p.get("name"),
                "Farm": p.get("farm_name"),
                "Lat": p.get("lat"),
                "Lng": p.get("lng"),
                "Width m": p.get("width_m"),
                "Height m": p.get("height_m"),
                "Map": p.get("map_type"),
                "Spot": p.get("spot_size"),
            }
            for p in st.session_state.map_profiles
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Settings Roadmap")
    st.caption("Next sections: camera quality, detection thresholds, map alerts, and historical season profiles.")


def _render_help_page():
    st.markdown("### Help")
    st.markdown(
        "1. Start live camera feed with **START**.\n"
        "2. Keep canopy centered and maintain stable movement.\n"
        "3. Use the **Settings** tab for display presets and farm location.\n"
        "4. Open map in precision mode when validating detection clusters."
    )
    st.markdown("### Demo Video")
    st.text_input("Demo video URL or local file path", key="demo_video_url", placeholder="https://... or recordings/demo.mp4")
    if st.session_state.demo_video_url.strip():
        try:
            st.video(st.session_state.demo_video_url.strip())
        except Exception as e:
            st.warning(f"Could not load demo video: {e}")
    else:
        st.info("Add a demo video URL/path above to display it here.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

_apply_pending_actions()

with st.sidebar:

    st.markdown(
        '<div style="text-align:center;padding:0.6rem 0 0.2rem 0">'
        '<span style="font-size:1.4rem">🌽</span>'
        '<span style="font-size:0.9rem;font-weight:700;color:#eaedf3;'
        'margin-left:8px;letter-spacing:-0.3px">Agri-Scout</span>'
        '<span style="font-size:0.55rem;color:#3d4250;margin-left:6px;'
        "font-family:'IBM Plex Mono',monospace\">v4.0</span>"
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Operator Mode</div>', unsafe_allow_html=True)
    st.toggle(
        "Tractor mode (large controls)",
        value=st.session_state.tractor_mode,
        key="tractor_mode",
    )
    st.toggle(
        "Show advanced telemetry",
        value=st.session_state.show_advanced_ui,
        key="show_advanced_ui",
    )
    st.toggle(
        "Sunlight readability boost",
        value=st.session_state.sunlight_mode,
        key="sunlight_mode",
    )
    st.caption("Display theme and farm location are now in the Settings tab.")

    # ---- Vehicle ----
    st.markdown('<div class="sidebar-section-label">Vehicle</div>', unsafe_allow_html=True)

    speed_val = sim.get_speed_mph() if sim else 5.0
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Ground Speed</div>'
        '<div class="tc-val">' + f"{speed_val:.1f}" + ' <span class="tc-unit">MPH</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Heading</div>'
        '<div class="tc-val">' + f"{tractor_heading:.0f}" + '° <span class="tc-unit">' + heading_cardinal + '</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Disease Detections Logged</div>'
        '<div class="tc-val">' + red_dot + str(disease_count) + '</div>'
        '<div class="tc-sub">Geo-tagged  •  5s debounce</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.show_advanced_ui:
        st.markdown(
            '<div class="sidebar-section-label">Navigation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">GPS Status</div>'
            '<div class="tc-val"><span class="live-dot green"></span>Active '
            '<span class="tc-unit">RTK Fix</span></div>'
            '<div class="tc-sub">Satellites: 14  •  HDOP: 0.8</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Position</div>'
            '<div class="tc-val" style="font-size:0.9rem">'
            + lat_str + '° ' + lat_dir + ' &nbsp; '
            + lng_str + '° ' + lng_dir +
            '</div>'
            '<div class="tc-sub">' + farm_name_display + '  •  Simulated</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-section-label">Compute</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Snapdragon X Elite NPU</div>'
            '<div class="tc-val" style="font-size:0.95rem">'
            '<span class="live-dot blue"></span>Online</div>'
            '<div class="tc-sub">45 TOPS  •  INT8 quantised  •  QNN 2.28</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Model</div>'
            '<div class="tc-val" style="font-size:0.85rem">MobileNetV2</div>'
            '<div class="tc-sub">ONNX opset 14  •  4 classes  •  9.2 MB</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Pipeline</div>'
            '<div class="tc-val" style="font-size:0.85rem">Async v2</div>'
            '<div class="tc-sub">WebRTC 30fps → Infer ~6fps → Vote ×7</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Active Field</div>'
            '<div class="tc-val" style="font-size:0.9rem">' + farm_name_display + '</div>'
            '<div class="tc-sub">Custom farm profile  •  Location configurable</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    session_time = datetime.datetime.now().strftime("%b %d, %Y — %H:%M")
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Session Started</div>'
        '<div class="tc-val" style="font-size:0.82rem">' + session_time + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    st.session_state.live_map_refresh = False
    if st.session_state.tractor_mode:
        st.caption("Quick mode: tap once to refresh map and status.")
    else:
        st.caption("Live auto-refresh is disabled to prevent map flicker. Use the refresh button below.")

    # ---- Refresh button (updates map + telemetry) ----
    if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
        st.rerun()

    st.markdown(
        '<div style="text-align:center;font-size:0.55rem;color:#2a2d38;'
        "font-family:'IBM Plex Mono',monospace;letter-spacing:0.8px;margin-top:8px\">"
        'QUALCOMM EDGE AI HACKATHON 2026</div>',
        unsafe_allow_html=True,
    )

st.markdown(_build_theme_override_css(), unsafe_allow_html=True)

if st.session_state.sunlight_mode:
    st.markdown(
        """
        <style>
        .tcard,.feed-wrap,.det-panel,.advisory-panel,.map-panel,.quick-card{border-width:2px !important}
        .det-class{font-size:1.3rem !important}
        .cab-alert-state{font-size:1.2rem !important}
        .feed-label,.tc-label,.quick-label{color:#98a1b7 !important}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_drive_page():
    label = st.session_state.last_class
    conf = st.session_state.last_conf
    count = st.session_state.frame_count

    is_healthy = "healthy" in label.lower()
    is_waiting = "waiting" in label.lower() or "initial" in label.lower()

    if is_waiting:
        op_level = "warn"
        op_state = "SCANNING"
        op_hint = "Keep camera on crop canopy."
    elif is_healthy:
        op_level = "safe"
        op_state = "ALL CLEAR"
        op_hint = "No treatment needed right now."
    else:
        op_level = "danger"
        op_state = "ACTION NEEDED"
        op_hint = "Review treatment advisory below."

    if st.session_state.tractor_mode:
        st.markdown(
            '<div class="cab-alert ' + op_level + '">'
            '<div class="cab-alert-top">'
            '<div>'
            '<div class="cab-alert-state">' + op_state + ' · ' + label + '</div>'
            '<div class="cab-alert-meta">Confidence ' + f"{conf * 100:.0f}" + '%  •  Detections ' + str(disease_count) + '</div>'
            '</div>'
            '<span class="cab-pill ' + op_level + '">' + op_state + '</span>'
            '</div>'
            '<div class="cab-hint">' + op_hint + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="quick-row">'
            '<div class="quick-card">'
            '<div class="quick-label">Recent Detection</div>'
            '<div class="quick-val">' + last_det_age_text + '</div>'
            '<div class="quick-sub">Distance: ' + last_det_dist_text + '</div>'
            '</div>'
            '<div class="quick-card">'
            '<div class="quick-label">Current Heading</div>'
            '<div class="quick-val">' + f"{tractor_heading:.0f}° {heading_cardinal}" + '</div>'
            '<div class="quick-sub">Speed: ' + f"{(sim.get_speed_mph() if sim else 5.0):.1f} MPH" + '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.tractor_mode:
        st.markdown(
            '<div class="hdr hdr-compact"><div class="hdr-left">'
            '<div class="hdr-logo">AS</div>'
            '<div><h1 style="font-size:1.1rem">Agri-Scout</h1></div>'
            '</div><div class="hdr-right">'
            '<span class="hdr-chip chip-live"><span class="live-dot green"></span> Live Scan</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hdr"><div class="hdr-left">'
            '<div class="hdr-logo">AS</div>'
            '<div><h1>Agri-Scout</h1>'
            '<div class="sub">Real-Time Crop Disease Detection System</div></div>'
            '</div><div class="hdr-right">'
            '<span class="hdr-chip chip-npu">Snapdragon X Elite</span>'
            '<span class="hdr-chip chip-live"><span class="live-dot green"></span> Live</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    show_status_panel = st.session_state.show_advanced_ui or not st.session_state.tractor_mode
    if show_status_panel:
        col_feed, col_status = st.columns([6, 2] if st.session_state.tractor_mode else [5, 2])
    else:
        col_feed = st.container()
        col_status = None

    with col_feed:
        st.markdown(
            '<div class="feed-label"><span class="live-dot green"></span> Live Camera Feed</div>',
            unsafe_allow_html=True,
        )

        ctx = webrtc_streamer(
            key="agri-scout-cam",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        if st.session_state.tractor_mode and not ctx.state.playing:
            st.caption("Tap START, then allow camera permission to begin live scanning.")

    if col_status is not None:
        with col_status:
            st.markdown('<div class="feed-label">Detection Status</div>', unsafe_allow_html=True)

            if is_waiting:
                state = "waiting"
                icon_char = "⏳"
            elif is_healthy:
                state = "healthy"
                icon_char = "✅"
            else:
                state = "disease"
                icon_char = "⚠️"

            conf_pct = f"{conf * 100:.1f}"
            conf_width = f"{conf * 100:.0f}"

            st.markdown(
                '<div class="det-panel ' + state + '">'
                '<div class="det-icon ' + state + '">' + icon_char + '</div>'
                '<div class="det-body">'
                '<div class="det-class">' + label + '</div>'
                '<div class="det-meta">Confidence: ' + conf_pct + '%  •  #' + str(count) + '</div>'
                '<div class="det-conf-bar">'
                '<div class="det-conf-fill ' + state + '" style="width:' + conf_width + '%"></div>'
                '</div></div></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="height:8px"></div>'
                '<div class="feed-label">System Metrics</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="tcard">'
                '<div class="tc-label">Inferences Completed</div>'
                '<div class="tc-val">' + str(count) + '</div>'
                '</div>'
                '<div class="tcard">'
                '<div class="tc-label">Smoothing Window</div>'
                '<div class="tc-val">7 <span class="tc-unit">frames</span></div>'
                '<div class="tc-sub">Majority vote temporal filter</div>'
                '</div>'
                '<div class="tcard">'
                '<div class="tc-label">Detection Mode</div>'
                '<div class="tc-val" style="font-size:0.85rem">'
                '<span class="live-dot green"></span>Continuous</div>'
                '<div class="tc-sub">Async pipeline  •  ~6 FPS inference</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    conf_pct = f"{conf * 100:.1f}"
    advisory_text = st.session_state.advisory_text
    if advisory_text:
        advisory_body_html = '<div class="adv-body"><div class="adv-content">' + advisory_text + '</div></div>'
    else:
        if is_waiting or is_healthy:
            placeholder = (
                '<span style="color:#6e7790">$</span> awaiting disease detection…'
                '<span class="cursor"></span><br>'
                '<span style="color:#7f889f">// Treatment plan appears after a confirmed disease detection.</span>'
            )
        else:
            placeholder = (
                '<span style="color:#ff6b6b">⚡</span> '
                '<span style="color:#8d96ac">Disease detected:</span> '
                '<span style="color:#eaedf3;font-weight:600">' + label + '</span> '
                '<span style="color:#8d96ac">at</span> '
                '<span style="color:#eaedf3">' + conf_pct + '%</span><br>'
                '<span style="color:#6e7790">$</span> generating treatment advisory…'
                '<span class="cursor"></span>'
            )
        advisory_body_html = '<div class="adv-body"><div class="adv-placeholder">' + placeholder + '</div></div>'

    advisory_footer_html = ""
    if st.session_state.show_advanced_ui or not st.session_state.tractor_mode:
        advisory_footer_html = (
            '<div class="adv-footer">'
            '<div class="adv-footer-item">Model: Claude / Llama 3</div>'
            '<div class="adv-footer-item">Context: Field conditions + USDA guidelines</div>'
            '<div class="adv-footer-item">Refresh: On detection change</div>'
            '</div>'
        )

    st.markdown(
        '<div class="advisory-panel">'
        '<div class="adv-header">'
        '<div class="adv-header-left">'
        '<div class="adv-icon">🧬</div>'
        '<span class="adv-title">AI Treatment Advisory</span></div>'
        '<span class="adv-status">Powered by LLM Agent</span></div>'
        + advisory_body_html +
        advisory_footer_html +
        '</div>',
        unsafe_allow_html=True,
    )

    map_ctrl_left, map_ctrl_mid, map_ctrl_lat, map_ctrl_lng, map_ctrl_apply = st.columns([2, 2, 2, 2, 1.3])
    with map_ctrl_left:
        st.toggle("Precision map pan/zoom", value=st.session_state.map_interactive, key="map_interactive")
    with map_ctrl_mid:
        st.selectbox(
            "Map layer",
            _MAP_TYPES,
            index=_MAP_TYPES.index(_coerce_map_type(st.session_state.map_base_type)),
            key="map_base_type",
        )
    with map_ctrl_lat:
        st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            step=0.0001,
            format="%.6f",
            key="drive_lat_input",
        )
    with map_ctrl_lng:
        st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            step=0.0001,
            format="%.6f",
            key="drive_lng_input",
        )
    with map_ctrl_apply:
        st.markdown('<div style="height:1.62rem"></div>', unsafe_allow_html=True)
        if st.button("Apply Coords", use_container_width=True, key="drive_apply_coords_btn"):
            st.session_state.pending_drive_coord_apply = True
            st.rerun()

    def _render_map_panel(map_lat, map_lng, map_heading, map_disease_count):
        sim_state = st.session_state.tractor_sim
        map_trail_count = len(sim_state.path_history) if sim_state is not None else 0
        map_height = 500 if st.session_state.tractor_mode else 420

        st.markdown('<div class="map-panel">', unsafe_allow_html=True)
        if _MAP_AVAILABLE and sim_state is not None:
            try:
                st.markdown(
                    '<div class="map-header">'
                    '<div class="map-header-left">'
                    '<div class="map-icon">🗺️</div>'
                    '<span class="map-title">Ghost Tractor — Field Map</span></div>'
                    '<div class="map-stats">'
                    '<span class="map-stat">Farm: ' + farm_name_display + '</span>'
                    '<span class="map-stat">Detections: <strong>' + str(map_disease_count) + '</strong></span>'
                    '<span class="map-stat">Trail: ' + str(map_trail_count) + ' pts</span>'
                    '<span class="map-stat">View: ' + ('Interactive' if st.session_state.map_interactive else 'Locked') + ' ' + st.session_state.map_base_type.title() + '</span>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
                map_html = generate_map_html(
                    api_key=GOOGLE_MAPS_API_KEY,
                    current_pos=(map_lat, map_lng),
                    heading=map_heading,
                    path_history=sim_state.path_history,
                    disease_logs=st.session_state.disease_logs,
                    field_center=sim_state.get_field_center(),
                    field_bounds=sim_state.get_field_bounds(),
                    map_height=map_height,
                    map_interactive=st.session_state.map_interactive,
                    map_zoom=st.session_state.map_zoom_level,
                    map_type=st.session_state.map_base_type,
                    spot_scale=st.session_state.disease_spot_size,
                )
                components.html(map_html, height=map_height, scrolling=False)
            except Exception as e:
                st.error(f"Map rendering error: {e}")
        else:
            st.warning(
                "**Map unavailable.** Make sure `map_engine.py` is in the same "
                "directory as `app.py` and restart Streamlit."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    _render_map_panel(
        tractor_lat,
        tractor_lng,
        tractor_heading,
        len(st.session_state.disease_logs),
    )

    if st.session_state.tractor_mode:
        st.markdown(
            '<div class="metrics-row">'
            '<div class="mcard"><div class="mc-label">Detections</div>'
            '<div class="mc-val" style="color:#ff6b6b">' + str(len(st.session_state.disease_logs)) + '</div></div>'
            '<div class="mcard"><div class="mc-label">Last Detection</div><div class="mc-val">' + last_det_age_text + '</div></div>'
            '<div class="mcard"><div class="mc-label">Distance</div><div class="mc-val">' + last_det_dist_text + '</div></div>'
            '<div class="mcard"><div class="mc-label">Heading</div><div class="mc-val">' + f"{tractor_heading:.0f}° {heading_cardinal}" + '</div></div>'
            '<div class="mcard"><div class="mc-label">Speed</div><div class="mc-val">' + f"{(sim.get_speed_mph() if sim else 5.0):.1f} MPH" + '</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.show_advanced_ui:
        st.markdown(
            '<div class="metrics-row">'
            '<div class="mcard"><div class="mc-label">Model</div><div class="mc-val">MobileNetV2</div></div>'
            '<div class="mcard"><div class="mc-label">Runtime</div><div class="mc-val">ONNX 14</div></div>'
            '<div class="mcard"><div class="mc-label">Target</div><div class="mc-val">SDX Elite</div></div>'
            '<div class="mcard"><div class="mc-label">Detections</div>'
            '<div class="mc-val" style="color:#ff6b6b">' + str(len(st.session_state.disease_logs)) + '</div></div>'
            '<div class="mcard"><div class="mc-label">Pipeline</div><div class="mc-val">Async v4</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metrics-row">'
            '<div class="mcard"><div class="mc-label">Detections</div><div class="mc-val">' + str(len(st.session_state.disease_logs)) + '</div></div>'
            '<div class="mcard"><div class="mc-label">Last Class</div><div class="mc-val">' + label + '</div></div>'
            '<div class="mcard"><div class="mc-label">Confidence</div><div class="mc-val">' + conf_pct + '%</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )


top_tab_drive, top_tab_data, top_tab_compute, top_tab_gallery, top_tab_settings, top_tab_help = st.tabs(
    ["🚜 Drive", "📊 Data Visualization", "⚙️ NPU/GPU/CPU Power", "🖼️ Gallery", "⚙️ Settings", "🆘 Help + Demo"]
)

with top_tab_drive:
    _render_drive_page()

with top_tab_data:
    _render_data_visualization_page()

with top_tab_compute:
    _render_compute_page()

with top_tab_gallery:
    _render_gallery_page()

with top_tab_settings:
    _render_settings_page()

with top_tab_help:
    _render_help_page()
# --- HEATMAP STRESS TESTER ---
if st.session_state.show_advanced_ui:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Dev Tools")
    if st.sidebar.button("🚨 Inject 10 Fake Detections"):
        if DiseaseLog is None or st.session_state.tractor_sim is None:
            st.sidebar.warning("Map simulator is unavailable; cannot inject test detections.")
        else:
            base_lat, base_lng, _ = st.session_state.tractor_sim.get_current_position()

            for _ in range(10):
                fake_lat = base_lat + random.uniform(-0.0005, 0.0005)
                fake_lng = base_lng + random.uniform(-0.0005, 0.0005)

                st.session_state.disease_logs.append(DiseaseLog(
                    lat=fake_lat,
                    lng=fake_lng,
                    label="Simulated Rust",
                    confidence=random.uniform(0.7, 0.99),
                    timestamp=time.time(),
                ))
            st.sidebar.success("Injected 10 points. Check the map!")
            st.rerun()
