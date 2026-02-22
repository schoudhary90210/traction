#!/usr/bin/env python3
"""
================================================================================
 TRACTION — Real-Time Crop Disease Detection Dashboard  (v4 — Ghost Tractor)
================================================================================
 Run with:   streamlit run app.py
================================================================================
"""

import csv
import datetime
import json
import math
import os
import platform
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from vision_engine import AgriScoutEngine

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import psutil
except Exception:
    psutil = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

_PROCESS = psutil.Process(os.getpid()) if psutil else None

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


def _load_maps_api_key():
    key = ""
    try:
        key = str(st.secrets.get("GOOGLE_MAPS_API_KEY", "")).strip()
    except Exception:
        key = ""
    if not key:
        key = str(os.getenv("GOOGLE_MAPS_API_KEY", "")).strip()
    return key


GOOGLE_MAPS_API_KEY = _load_maps_api_key()

_LOG_COOLDOWN_SEC = 5.0
_LOG_MIN_DISTANCE_M = 8.0
_DISEASE_CONFIDENCE_THRESH = 0.85
_HIGH_DETECTION_CAPTURE_THRESH = 0.92
_LIVE_SAMPLE_LOG_INTERVAL_SEC = 1.0
_VIDEO_RECORD_FPS = 12.0
_COLOR_SCHEMES = [
    "Night Field (Dark)",
    "Daylight Paper (Black on Light)",
    "Green Cabin (White on Dark Green)",
    "Amber Signal (White on Amber)",
    "Skyline Blue (White on Deep Blue)",
    "Dust Gray (Black on Soft Gray)",
    "Field Sunrise (Black on Light Yellow)",
]
_LEGACY_COLOR_SCHEME_MAP = {
    "Amber Contrast (Black on Amber)": "Amber Signal (White on Amber)",
}
_DEFAULT_FARM_NAME = "North-40"
_DEFAULT_FARM_LAT = globals().get("_DEFAULT_FARM_LAT", 43.1580)
_DEFAULT_FARM_LNG = globals().get("_DEFAULT_FARM_LNG", -89.9120)
_DEFAULT_FARM_WIDTH_M = globals().get("_DEFAULT_FARM_WIDTH_M", 800.0)
_DEFAULT_FARM_HEIGHT_M = globals().get("_DEFAULT_FARM_HEIGHT_M", 708.0)
_DEFAULT_DISEASE_SPOT_SIZE = 10.0
_DETECTION_DATASET_ROOT = Path("dataset") / "detections"
_LIVE_SESSIONS_ROOT = Path("dataset") / "live_sessions"
_MAP_TYPES = ["roadmap", "hybrid", "satellite", "terrain"]
_APP_LOGO_PATH = Path("assets") / "traction-logo.svg"


# ==============================================================================
#  Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="TRACTION | Qualcomm Edge AI",
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
.app-title-wrap{padding:0.2rem 0 0.7rem 0}
.app-title-main{font-size:2rem;font-weight:800;color:#eaedf3;letter-spacing:-0.6px;line-height:1.05}
.app-title-sub{font-size:0.76rem;color:#7e879c;text-transform:uppercase;letter-spacing:1.4px;margin-top:4px}
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
.advisory-panel::before{display:none}
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
.map-panel::before{display:none}
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
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ==============================================================================
#  Singleton Engine
# ==============================================================================

@st.cache_resource(show_spinner="Loading TRACTION model…")
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
    st.session_state.map_base_type = "satellite"
if st.session_state.map_base_type not in _MAP_TYPES:
    st.session_state.map_base_type = "satellite"
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
if st.session_state.color_scheme in _LEGACY_COLOR_SCHEME_MAP:
    st.session_state.color_scheme = _LEGACY_COLOR_SCHEME_MAP[st.session_state.color_scheme]
if st.session_state.color_scheme not in _COLOR_SCHEMES:
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
    raw_map_type = str(profile.get("map_type", "satellite")).lower().strip()
    profile["map_type"] = raw_map_type if raw_map_type in _MAP_TYPES else "satellite"
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
    st.session_state.profile_map_type = "satellite"
if "settings_notice" not in st.session_state:
    st.session_state.settings_notice = ""
if "demo_video_url" not in st.session_state:
    st.session_state.demo_video_url = ""
if "gallery_limit" not in st.session_state:
    st.session_state.gallery_limit = 24
if "pending_reset_farm" not in st.session_state:
    st.session_state.pending_reset_farm = False
if "pending_apply_farm" not in st.session_state:
    st.session_state.pending_apply_farm = False
if "pending_save_profile_name" not in st.session_state:
    st.session_state.pending_save_profile_name = ""
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
if "drive_selected_profile_name" not in st.session_state:
    st.session_state.drive_selected_profile_name = st.session_state.selected_profile_name
if "sidebar_quick_lat" not in st.session_state:
    st.session_state.sidebar_quick_lat = float(st.session_state.active_farm_lat)
if "sidebar_quick_lng" not in st.session_state:
    st.session_state.sidebar_quick_lng = float(st.session_state.active_farm_lng)
if "sidebar_profile_choice" not in st.session_state:
    st.session_state.sidebar_profile_choice = st.session_state.selected_profile_name
if "app_start_time" not in st.session_state:
    st.session_state.app_start_time = time.time()
if "compute_auto_refresh" not in st.session_state:
    st.session_state.compute_auto_refresh = False
if "camera_prev_playing" not in st.session_state:
    st.session_state.camera_prev_playing = False
if "active_recording_id" not in st.session_state:
    st.session_state.active_recording_id = ""
if "active_recording_started_at" not in st.session_state:
    st.session_state.active_recording_started_at = ""
if "active_recording_event_count" not in st.session_state:
    st.session_state.active_recording_event_count = 0
if "active_recording_capture_count" not in st.session_state:
    st.session_state.active_recording_capture_count = 0
if "active_recording_frame_count" not in st.session_state:
    st.session_state.active_recording_frame_count = 0
if "active_recording_video_path" not in st.session_state:
    st.session_state.active_recording_video_path = ""
if "last_live_sample_log_ts" not in st.session_state:
    st.session_state.last_live_sample_log_ts = 0.0
if "recording_notice" not in st.session_state:
    st.session_state.recording_notice = ""
if "sidebar_session_choice" not in st.session_state:
    st.session_state.sidebar_session_choice = ""


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  THREADING / INFERENCE — 100% UNCHANGED FROM v2                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

_frame_lock = threading.Lock()
_frame_slot: Optional[np.ndarray] = None
_frame_event = threading.Event()
_capture_lock = threading.Lock()
_latest_frame_bgr: Optional[np.ndarray] = None
_recording_lock = threading.Lock()
_recording_writer = None
_recording_video_path: str = ""
_recording_video_size = (0, 0)

_result_lock = threading.Lock()
_result_label: str = "Initialising…"
_result_conf: float = 0.0
_result_latency_ms: float = 0.0
_inference_count: int = 0
_latency_history_ms = deque(maxlen=300)
_inference_started_at = time.time()
_last_infer_ts = 0.0

_INFERENCE_COOLDOWN = 0.15
_worker_running = False


def _inference_worker():
    global _result_label, _result_conf, _result_latency_ms, _inference_count, _last_infer_ts
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
            _latency_history_ms.append(latency)
            _last_infer_ts = time.time()

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

    cv2.putText(frame, "TRACTION  v4", (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 160, 120), 1, cv2.LINE_AA)

    return frame


# ==============================================================================
#  WebRTC Callback — UNCHANGED
# ==============================================================================

_FRAME_SKIP = 5
_callback_counter: int = 0


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global _frame_slot, _callback_counter, _latest_frame_bgr

    img = frame.to_ndarray(format="bgr24")
    with _capture_lock:
        _latest_frame_bgr = img.copy()

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
    if st.session_state.active_recording_id:
        _record_live_video_frame(img)
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


def _is_disease_detection(det_label, det_conf):
    det_label_lower = str(det_label).lower()
    return (
        "healthy" not in det_label_lower
        and "waiting" not in det_label_lower
        and "initial" not in det_label_lower
        and float(det_conf) >= _DISEASE_CONFIDENCE_THRESH
    )


def _farm_slug(raw_name):
    slug_chars = []
    for ch in str(raw_name).strip().lower():
        if ch.isalnum():
            slug_chars.append(ch)
        elif ch in {" ", "-", "_"}:
            slug_chars.append("-")
    slug = "".join(slug_chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return slug or "farm"


def _safe_map_type_value(raw_map_type):
    map_type = str(raw_map_type).lower().strip()
    return map_type if map_type in _MAP_TYPES else "satellite"


def _append_detection_dataset_row(lat, lng, label, confidence, coord_source, timestamp):
    farm_name = (st.session_state.active_farm_name or _DEFAULT_FARM_NAME).strip() or _DEFAULT_FARM_NAME
    farm_slug = _farm_slug(farm_name)
    ts = datetime.datetime.fromtimestamp(float(timestamp))
    date_str = ts.strftime("%Y-%m-%d")
    csv_dir = _DETECTION_DATASET_ROOT / farm_slug
    csv_path = csv_dir / f"{date_str}.csv"

    fieldnames = [
        "timestamp",
        "date",
        "time",
        "farm_name",
        "farm_slug",
        "disease_label",
        "confidence_pct",
        "latitude",
        "longitude",
        "coord_source",
        "map_layer",
    ]
    row = {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": date_str,
        "time": ts.strftime("%H:%M:%S"),
        "farm_name": farm_name,
        "farm_slug": farm_slug,
        "disease_label": str(label),
        "confidence_pct": round(float(confidence) * 100.0, 2),
        "latitude": round(float(lat), 7),
        "longitude": round(float(lng), 7),
        "coord_source": str(coord_source),
        "map_layer": _safe_map_type_value(st.session_state.map_base_type),
    }

    try:
        csv_dir.mkdir(parents=True, exist_ok=True)
        write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        # Dataset logging should never break live detection UI.
        pass


def _live_session_events_fieldnames():
    return [
        "timestamp",
        "date",
        "time",
        "session_id",
        "farm_name",
        "event_type",
        "disease_label",
        "confidence_pct",
        "latitude",
        "longitude",
        "coord_source",
        "map_layer",
        "color_scheme",
        "high_contrast_mode",
        "invert_colors_mode",
        "sunlight_mode",
        "snapshot_path",
        "note",
    ]


def _session_dir(session_id):
    return _LIVE_SESSIONS_ROOT / session_id


def _session_events_csv_path(session_id):
    return _session_dir(session_id) / "events.csv"


def _session_meta_path(session_id):
    return _session_dir(session_id) / "meta.json"


def _read_json(path, default_value):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def _write_json(path, payload):
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _next_session_id():
    base = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base
    n = 1
    while (_LIVE_SESSIONS_ROOT / candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _release_recording_writer():
    global _recording_writer, _recording_video_path, _recording_video_size
    writer_to_close = None
    with _recording_lock:
        if _recording_writer is not None:
            writer_to_close = _recording_writer
        _recording_writer = None
        _recording_video_path = ""
        _recording_video_size = (0, 0)
    if writer_to_close is not None:
        try:
            writer_to_close.release()
        except Exception:
            pass


def _open_video_writer(path, width, height):
    target_path = str(path)
    if width <= 0 or height <= 0:
        return None, ""

    codec_plan = [
        ("mp4v", target_path),
        ("avc1", target_path),
        ("MJPG", str(Path(target_path).with_suffix(".avi"))),
    ]

    for codec, attempt_path in codec_plan:
        try:
            Path(attempt_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                attempt_path,
                cv2.VideoWriter_fourcc(*codec),
                float(_VIDEO_RECORD_FPS),
                (int(width), int(height)),
            )
            if writer is not None and writer.isOpened():
                return writer, attempt_path
            if writer is not None:
                writer.release()
        except Exception:
            continue
    return None, ""


def _record_live_video_frame(frame_bgr):
    global _recording_writer, _recording_video_path, _recording_video_size

    session_id = str(st.session_state.active_recording_id or "").strip()
    target_path = str(st.session_state.active_recording_video_path or "").strip()
    if not session_id or not target_path:
        return False

    if frame_bgr is None:
        return False
    try:
        h, w = frame_bgr.shape[:2]
    except Exception:
        return False
    if h <= 0 or w <= 0:
        return False

    writer_to_release = None
    with _recording_lock:
        needs_reopen = (
            _recording_writer is None
            or _recording_video_path != target_path
            or _recording_video_size != (int(w), int(h))
        )
        if needs_reopen:
            writer_to_release = _recording_writer
            _recording_writer = None
            _recording_video_path = ""
            _recording_video_size = (0, 0)

            writer, resolved_path = _open_video_writer(target_path, int(w), int(h))
            if writer is None:
                if writer_to_release is not None:
                    try:
                        writer_to_release.release()
                    except Exception:
                        pass
                return False
            _recording_writer = writer
            _recording_video_path = resolved_path
            _recording_video_size = (int(w), int(h))
            st.session_state.active_recording_video_path = resolved_path

        try:
            _recording_writer.write(frame_bgr)
        except Exception:
            return False

    if writer_to_release is not None:
        try:
            writer_to_release.release()
        except Exception:
            pass

    st.session_state.active_recording_frame_count = int(st.session_state.active_recording_frame_count) + 1
    return True


def _start_live_recording_session():
    _release_recording_writer()
    session_id = _next_session_id()
    session_dir = _session_dir(session_id)
    capture_dir = session_dir / "captures"
    capture_dir_mirror = Path("captures") / "live_sessions" / session_id
    video_dir_mirror = Path("recordings") / "live_sessions" / session_id
    video_path = session_dir / "video.mp4"
    video_path_mirror = video_dir_mirror / "video.mp4"
    events_path = _session_events_csv_path(session_id)
    started_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_dir_mirror.mkdir(parents=True, exist_ok=True)
        video_dir_mirror.mkdir(parents=True, exist_ok=True)
        with events_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_live_session_events_fieldnames())
            writer.writeheader()
    except Exception as e:
        st.session_state.recording_notice = f"Could not start recording session: {e}"
        return False

    meta = {
        "session_id": session_id,
        "farm_name": (st.session_state.active_farm_name or _DEFAULT_FARM_NAME).strip() or _DEFAULT_FARM_NAME,
        "status": "active",
        "started_at": started_at,
        "ended_at": "",
        "events_csv": str(events_path),
        "video_path": str(video_path),
        "video_mirror_path": str(video_path_mirror),
        "video_status": "pending",
        "video_fps": float(_VIDEO_RECORD_FPS),
        "video_frame_count": 0,
    }
    _write_json(_session_meta_path(session_id), meta)

    st.session_state.active_recording_id = session_id
    st.session_state.active_recording_started_at = started_at
    st.session_state.active_recording_event_count = 0
    st.session_state.active_recording_capture_count = 0
    st.session_state.active_recording_frame_count = 0
    st.session_state.active_recording_video_path = str(video_path)
    st.session_state.last_live_sample_log_ts = 0.0
    st.session_state.recording_notice = f"Recording session started: {session_id}"
    st.session_state.sidebar_session_choice = session_id
    return True


def _stop_live_recording_session():
    session_id = st.session_state.active_recording_id
    if not session_id:
        return False

    video_path = str(st.session_state.active_recording_video_path or "").strip()
    frame_count = int(st.session_state.active_recording_frame_count)
    _release_recording_writer()

    video_mirror_path = str(Path("recordings") / "live_sessions" / session_id / Path(video_path).name) if video_path else ""
    copied_video = False
    if video_path and Path(video_path).exists():
        if video_mirror_path:
            try:
                Path(video_mirror_path).parent.mkdir(parents=True, exist_ok=True)
                if Path(video_mirror_path).resolve() != Path(video_path).resolve():
                    shutil.copy2(video_path, video_mirror_path)
                copied_video = True
            except Exception:
                copied_video = False

    meta_path = _session_meta_path(session_id)
    meta = _read_json(meta_path, {})
    if not isinstance(meta, dict):
        meta = {}
    meta["session_id"] = session_id
    meta["farm_name"] = meta.get("farm_name") or (st.session_state.active_farm_name or _DEFAULT_FARM_NAME)
    meta["status"] = "stopped"
    meta["ended_at"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if video_path:
        meta["video_path"] = video_path
    if video_mirror_path:
        meta["video_mirror_path"] = video_mirror_path
    meta["video_frame_count"] = frame_count
    meta["video_fps"] = float(_VIDEO_RECORD_FPS)
    if video_path and Path(video_path).exists():
        try:
            meta["video_size_kb"] = round(Path(video_path).stat().st_size / 1024.0, 1)
            meta["video_status"] = "ready"
        except Exception:
            meta["video_status"] = "unknown"
    else:
        meta["video_status"] = "missing"
    if copied_video:
        meta["video_mirrored"] = True
    _write_json(meta_path, meta)

    st.session_state.recording_notice = f"Recording session ended: {session_id}"
    st.session_state.active_recording_id = ""
    st.session_state.active_recording_started_at = ""
    st.session_state.active_recording_event_count = 0
    st.session_state.active_recording_capture_count = 0
    st.session_state.active_recording_frame_count = 0
    st.session_state.active_recording_video_path = ""
    st.session_state.last_live_sample_log_ts = 0.0
    return True


def _capture_detection_snapshot(label, confidence, timestamp):
    session_id = st.session_state.active_recording_id
    if not session_id:
        return ""

    with _capture_lock:
        frame = None if _latest_frame_bgr is None else _latest_frame_bgr.copy()
    if frame is None:
        return ""

    safe_label = _farm_slug(label).replace("-", "_")
    ts = datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_label}_{int(float(confidence) * 100)}.jpg"
    rel_path = _session_dir(session_id) / "captures" / filename
    mirror_path = Path("captures") / "live_sessions" / session_id / filename

    try:
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        ok_a = cv2.imwrite(str(rel_path), frame)
        ok_b = cv2.imwrite(str(mirror_path), frame)
        if ok_a and ok_b:
            st.session_state.active_recording_capture_count = int(st.session_state.active_recording_capture_count) + 1
            return str(rel_path)
    except Exception:
        return ""
    return ""


def _append_live_session_event(event_type, label, confidence, lat, lng, coord_source, snapshot_path="", note=""):
    session_id = st.session_state.active_recording_id
    if not session_id:
        return False

    events_path = _session_events_csv_path(session_id)
    now = datetime.datetime.now()
    row = {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "session_id": session_id,
        "farm_name": (st.session_state.active_farm_name or _DEFAULT_FARM_NAME).strip() or _DEFAULT_FARM_NAME,
        "event_type": str(event_type),
        "disease_label": str(label),
        "confidence_pct": round(float(confidence) * 100.0, 2),
        "latitude": round(float(lat), 7),
        "longitude": round(float(lng), 7),
        "coord_source": str(coord_source),
        "map_layer": _safe_map_type_value(st.session_state.map_base_type),
        "color_scheme": str(st.session_state.color_scheme),
        "high_contrast_mode": bool(st.session_state.high_contrast_mode),
        "invert_colors_mode": bool(st.session_state.invert_colors_mode),
        "sunlight_mode": bool(st.session_state.sunlight_mode),
        "snapshot_path": str(snapshot_path),
        "note": str(note),
    }
    try:
        write_header = (not events_path.exists()) or events_path.stat().st_size == 0
        with events_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_live_session_events_fieldnames())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        return False
    st.session_state.active_recording_event_count = int(st.session_state.active_recording_event_count) + 1
    return True


def _sync_live_recording_state(is_playing):
    was_playing = bool(st.session_state.camera_prev_playing)
    playing = bool(is_playing)
    if playing and not was_playing:
        _start_live_recording_session()
    elif was_playing and not playing:
        _stop_live_recording_session()
    st.session_state.camera_prev_playing = playing


def _log_live_inference_sample(lat, lng, coord_source):
    if not st.session_state.active_recording_id:
        return False
    now = time.time()
    if (now - float(st.session_state.last_live_sample_log_ts)) < _LIVE_SAMPLE_LOG_INTERVAL_SEC:
        return False
    ok = _append_live_session_event(
        event_type="inference",
        label=st.session_state.last_class,
        confidence=st.session_state.last_conf,
        lat=lat,
        lng=lng,
        coord_source=coord_source,
    )
    if ok:
        st.session_state.last_live_sample_log_ts = now
    return ok


def _list_live_recording_sessions(limit=120):
    sessions = []
    root = _LIVE_SESSIONS_ROOT
    if not root.exists():
        return sessions

    for d in root.iterdir():
        if not d.is_dir():
            continue
        sid = d.name
        meta = _read_json(d / "meta.json", {})
        if not isinstance(meta, dict):
            meta = {}

        events_csv = d / "events.csv"
        events_count = 0
        if events_csv.exists():
            try:
                with events_csv.open("r", encoding="utf-8") as f:
                    events_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                events_count = 0
        capture_dir = d / "captures"
        captures_count = 0
        if capture_dir.exists():
            try:
                captures_count = sum(1 for p in capture_dir.iterdir() if p.is_file())
            except Exception:
                captures_count = 0

        started_at = str(meta.get("started_at", ""))
        if not started_at:
            try:
                started_at = datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                started_at = ""

        video_candidates = [
            str(meta.get("video_mirror_path", "")).strip(),
            str(meta.get("video_path", "")).strip(),
            str((Path("recordings") / "live_sessions" / sid / "video.mp4")),
            str(d / "video.mp4"),
            str(d / "video.avi"),
        ]
        video_path_existing = ""
        for cand in video_candidates:
            if cand and Path(cand).exists():
                video_path_existing = cand
                break
        video_size_kb = 0.0
        if video_path_existing:
            try:
                video_size_kb = round(Path(video_path_existing).stat().st_size / 1024.0, 1)
            except Exception:
                video_size_kb = 0.0
        try:
            video_frame_count = int(float(meta.get("video_frame_count", 0) or 0))
        except Exception:
            video_frame_count = 0

        sessions.append(
            {
                "session_id": sid,
                "farm_name": str(meta.get("farm_name", _DEFAULT_FARM_NAME)),
                "status": str(meta.get("status", "unknown")),
                "started_at": started_at,
                "ended_at": str(meta.get("ended_at", "")),
                "events_count": int(events_count),
                "captures_count": int(captures_count),
                "video_path": video_path_existing,
                "video_status": str(meta.get("video_status", "")),
                "video_frame_count": int(video_frame_count),
                "video_size_kb": float(video_size_kb),
            }
        )

    sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return sessions[: int(max(1, limit))]


def _resolve_session_video_path(session_id, session_meta=None):
    sid = str(session_id).strip()
    if not sid:
        return ""
    if isinstance(session_meta, dict):
        existing = str(session_meta.get("video_path", "")).strip()
        if existing and Path(existing).exists():
            return existing
    candidates = [
        str(Path("recordings") / "live_sessions" / sid / "video.mp4"),
        str(Path("recordings") / "live_sessions" / sid / "video.avi"),
        str(_session_dir(sid) / "video.mp4"),
        str(_session_dir(sid) / "video.avi"),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    return ""


def _read_live_session_events(session_id, limit_rows=1500):
    sid = str(session_id).strip()
    if not sid:
        return []
    path = _session_events_csv_path(sid)
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= int(max(1, limit_rows)):
                    break
    except Exception:
        return []
    return rows


def _list_detection_dataset_files(limit=200):
    files = []
    root = _DETECTION_DATASET_ROOT
    if not root.exists():
        return files
    for p in root.rglob("*.csv"):
        if not p.is_file():
            continue
        farm_slug = p.parent.name
        date_part = p.stem
        files.append(
            {
                "farm_slug": farm_slug,
                "date": date_part,
                "path": str(p),
                "mtime": p.stat().st_mtime,
            }
        )
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files[: int(max(1, limit))]


def _read_detection_dataset_rows(csv_path, limit_rows=2000):
    path = Path(csv_path)
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= int(max(1, limit_rows)):
                    break
    except Exception:
        return []
    return rows


def _append_detection_event(lat, lng, label, confidence, timestamp, coord_source):
    if DiseaseLog is not None:
        event = DiseaseLog(
            lat=float(lat),
            lng=float(lng),
            label=str(label),
            confidence=float(confidence),
            timestamp=float(timestamp),
        )
    else:
        event = SimpleNamespace(
            lat=float(lat),
            lng=float(lng),
            label=str(label),
            confidence=float(confidence),
            timestamp=float(timestamp),
        )

    st.session_state.disease_logs.append(event)
    st.session_state.advisory_text = (
        f"{str(label)} detected on {st.session_state.active_farm_name} "
        f"at {float(confidence) * 100.0:.1f}% confidence."
    )
    _append_detection_dataset_row(
        lat=float(lat),
        lng=float(lng),
        label=label,
        confidence=float(confidence),
        coord_source=coord_source,
        timestamp=float(timestamp),
    )


def _build_placeholder_detection_coords():
    base_lat = float(st.session_state.active_farm_lat)
    base_lng = float(st.session_state.active_farm_lng)
    field_w_m = max(40.0, float(st.session_state.active_farm_width_m))
    field_h_m = max(40.0, float(st.session_state.active_farm_height_m))

    lat_span = max(0.00006, (field_h_m / 111_320.0) * 0.18)
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(base_lat))
    if abs(m_per_deg_lng) < 1e-6:
        m_per_deg_lng = 1.0
    lng_span = max(0.00006, (field_w_m / abs(m_per_deg_lng)) * 0.18)

    seed = int(time.time() * 1000.0) + int(st.session_state.frame_count)
    rng = random.Random(seed)
    lat = max(-85.0, min(85.0, base_lat + rng.uniform(-lat_span, lat_span)))
    lng = max(-179.9999, min(179.9999, base_lng + rng.uniform(-lng_span, lng_span)))
    return lat, lng


def _attempt_detection_log(lat, lng, coord_source):
    if not bool(st.session_state.camera_prev_playing):
        return False
    det_label = st.session_state.last_class
    det_conf = st.session_state.last_conf
    if not _is_disease_detection(det_label, det_conf):
        return False

    now = time.time()
    if (now - st.session_state.last_log_time) < _LOG_COOLDOWN_SEC:
        return False

    dist_ok = True
    logs = st.session_state.disease_logs
    if logs:
        last = logs[-1]
        dist_ok = _haversine_m(last.lat, last.lng, lat, lng) >= _LOG_MIN_DISTANCE_M
    if not dist_ok:
        return False

    _append_detection_event(
        lat=lat,
        lng=lng,
        label=det_label,
        confidence=det_conf,
        timestamp=now,
        coord_source=coord_source,
    )
    snapshot_rel_path = ""
    if float(det_conf) >= _HIGH_DETECTION_CAPTURE_THRESH:
        snapshot_rel_path = _capture_detection_snapshot(
            label=det_label,
            confidence=det_conf,
            timestamp=now,
        )
    _append_live_session_event(
        event_type="detection",
        label=det_label,
        confidence=det_conf,
        lat=lat,
        lng=lng,
        coord_source=coord_source,
        snapshot_path=snapshot_rel_path,
        note="auto-detection",
    )
    st.session_state.last_log_time = now
    return True


def _update_tractor_and_logs(simulator):
    """Advance tractor simulation and append disease logs with debounce."""
    tractor_lat, tractor_lng, tractor_heading = simulator.get_current_position()

    _CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    heading_cardinal = _CARDINALS[int(((tractor_heading + 22.5) % 360) / 45)]

    _attempt_detection_log(tractor_lat, tractor_lng, "tractor_sim")

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
    placeholder_lat, placeholder_lng = _build_placeholder_detection_coords()
    _attempt_detection_log(placeholder_lat, placeholder_lng, "placeholder")
    disease_count = len(st.session_state.disease_logs)

# Pre-build HTML snippets (avoids escaped-quote issues in f-strings)
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
            "text": "#0e0e0e",
            "muted": "#232323",
            "border": "#7f7450",
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
        "Amber Signal (White on Amber)": {
            "app_bg": "#3b2200",
            "panel": "#7a4700",
            "panel_alt": "#8a5000",
            "sidebar": "#693d00",
            "text": "#fffdf6",
            "muted": "#ffe4b0",
            "border": "#f2b141",
            "accent": "#fff5d9",
            "danger": "#ffd0d0",
        },
        "Amber Contrast (Black on Amber)": {
            "app_bg": "#3b2200",
            "panel": "#7a4700",
            "panel_alt": "#8a5000",
            "sidebar": "#693d00",
            "text": "#fffdf6",
            "muted": "#ffe4b0",
            "border": "#f2b141",
            "accent": "#fff5d9",
            "danger": "#ffd0d0",
        },
        "Skyline Blue (White on Deep Blue)": {
            "app_bg": "#081629",
            "panel": "#102845",
            "panel_alt": "#0d2036",
            "sidebar": "#0b1a2c",
            "text": "#f3f8ff",
            "muted": "#c8d6ea",
            "border": "#2b4e78",
            "accent": "#7dd3ff",
            "danger": "#ffb4b4",
        },
        "Dust Gray (Black on Soft Gray)": {
            "app_bg": "#f2f2ee",
            "panel": "#fbfbf8",
            "panel_alt": "#ecece6",
            "sidebar": "#e4e3dc",
            "text": "#151515",
            "muted": "#2f2f2f",
            "border": "#8b8b7c",
            "accent": "#2f6b3d",
            "danger": "#8d2222",
        },
        "Field Sunrise (Black on Light Yellow)": {
            "app_bg": "#fff8d6",
            "panel": "#fffef2",
            "panel_alt": "#fff4b8",
            "sidebar": "#ffeaa0",
            "text": "#111111",
            "muted": "#2b2b2b",
            "border": "#9f8a3f",
            "accent": "#165a3b",
            "danger": "#8b1a1a",
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
        app_luma = _luma(_hex_to_rgb(p["app_bg"]))
        contrast_text = "#000000" if app_luma >= 0.5 else "#ffffff"
        contrast_muted = contrast_text
        high_contrast_css = """
        .stApp,.stApp *{
          color:var(--ag-text) !important;
          -webkit-text-fill-color:var(--ag-text) !important;
        }
        """
    else:
        contrast_text = p["text"]
        contrast_muted = p["muted"]
        high_contrast_css = ""
    border_width = "2px" if st.session_state.high_contrast_mode else "1px"

    invert_css = ""
    if st.session_state.invert_colors_mode:
        invert_css = ".stApp{filter:invert(1) hue-rotate(180deg) !important;}"

    base_text_css = """
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp label,
    .stApp legend,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
      color: var(--ag-text) !important;
    }
    .stApp [data-testid="stCaptionContainer"] p {
      color: var(--ag-muted) !important;
    }
    .stApp .stTabs [data-baseweb="tab"],
    .stApp .stTabs [data-baseweb="tab"] * {
      color: var(--ag-text) !important;
    }
    .stApp .stTabs [aria-selected="true"] {
      border-bottom-color: var(--ag-accent) !important;
    }
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="textarea"] > div {
      background: var(--ag-panel) !important;
      border-color: var(--ag-border) !important;
      color: var(--ag-text) !important;
    }
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] div,
    .stApp input,
    .stApp textarea {
      color: var(--ag-text) !important;
      -webkit-text-fill-color: var(--ag-text) !important;
    }
    .stApp ::placeholder {
      color: var(--ag-muted) !important;
      opacity: 0.95 !important;
    }
    .stApp [data-testid="stToolbar"] *,
    .stApp [data-testid="collapsedControl"] * {
      color: var(--ag-text) !important;
    }
    .stApp hr,
    .stApp [data-testid="stMarkdownContainer"] hr {
      border: 0 !important;
      border-top: 2px solid var(--ag-border) !important;
      opacity: 1 !important;
    }
    .stApp [data-baseweb="divider"] {
      background-color: var(--ag-border) !important;
      opacity: 1 !important;
    }
    .stApp button[role="switch"] {
      background: var(--ag-panel-alt) !important;
      border: 2px solid var(--ag-border) !important;
      box-shadow: none !important;
    }
    .stApp button[role="switch"] > div {
      background: var(--ag-panel) !important;
      border: 1px solid var(--ag-border) !important;
    }
    .stApp button[role="switch"][aria-checked="true"] {
      background: var(--ag-accent) !important;
      border-color: var(--ag-accent) !important;
    }
    .stApp button[role="switch"][aria-checked="true"] > div {
      background: var(--ag-app-bg) !important;
      border-color: var(--ag-app-bg) !important;
    }
    .stApp [data-testid="stToggle"] p {
      color: var(--ag-text) !important;
    }
    """

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
    {base_text_css}
    {high_contrast_css}
    {invert_css}
    </style>
    """


def _coerce_map_type(raw_map_type):
    return _safe_map_type_value(raw_map_type)


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
    st.session_state.sidebar_quick_lat = lat
    st.session_state.sidebar_quick_lng = lng

    # Automatically keep the farm in the saved farms list (map profiles).
    auto_profile_name = st.session_state.active_farm_name
    auto_profile = {
        "name": auto_profile_name,
        "farm_name": auto_profile_name,
        "lat": float(lat),
        "lng": float(lng),
        "width_m": float(field_w_m),
        "height_m": float(field_h_m),
        "map_type": _coerce_map_type(st.session_state.map_base_type),
        "spot_size": float(st.session_state.disease_spot_size),
    }
    replaced = False
    for i, existing in enumerate(st.session_state.map_profiles):
        if existing.get("name") == auto_profile_name:
            st.session_state.map_profiles[i] = auto_profile
            replaced = True
            break
    if not replaced:
        st.session_state.map_profiles.append(auto_profile)
    st.session_state.selected_profile_name = auto_profile_name
    st.session_state.profile_name_input = auto_profile_name
    st.session_state.sidebar_profile_choice = auto_profile_name
    st.session_state.drive_selected_profile_name = auto_profile_name

    st.session_state.disease_logs = []
    st.session_state.last_log_time = 0.0
    return True, "Farm location updated. Map and tractor path reset. Farm profile saved."


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
    st.session_state.sidebar_profile_choice = name
    st.session_state.drive_selected_profile_name = name
    return True, "Profile saved."


def _load_profile_by_name(profile_name):
    for p in st.session_state.map_profiles:
        if p.get("name") == profile_name:
            loaded_map_type = _coerce_map_type(p.get("map_type", "satellite"))
            loaded_spot_size = float(max(3.0, min(24.0, float(p.get("spot_size", _DEFAULT_DISEASE_SPOT_SIZE)))))
            st.session_state.selected_profile_name = profile_name
            st.session_state.profile_name_input = profile_name
            st.session_state.sidebar_profile_choice = profile_name
            st.session_state.drive_selected_profile_name = profile_name
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
    st.session_state.sidebar_profile_choice = st.session_state.selected_profile_name
    st.session_state.drive_selected_profile_name = st.session_state.selected_profile_name
    st.session_state.profile_map_type = _coerce_map_type(first.get("map_type", st.session_state.map_base_type))
    return True, f"Deleted profile '{profile_name}'."


def _apply_pending_actions():
    save_name = st.session_state.get("pending_save_profile_name", "")
    if save_name:
        ok, msg = _save_current_profile(save_name)
        st.session_state.settings_notice = msg
        st.session_state.pending_save_profile_name = ""

    if st.session_state.get("pending_apply_farm", False):
        ok, msg = _apply_farm_location()
        st.session_state.settings_notice = msg
        st.session_state.pending_apply_farm = False

    if st.session_state.get("pending_reset_farm", False):
        st.session_state.farm_name = _DEFAULT_FARM_NAME
        st.session_state.farm_lat = float(_DEFAULT_FARM_LAT)
        st.session_state.farm_lng = float(_DEFAULT_FARM_LNG)
        st.session_state.farm_width_m = float(_DEFAULT_FARM_WIDTH_M)
        st.session_state.farm_height_m = float(_DEFAULT_FARM_HEIGHT_M)
        st.session_state.map_base_type = "satellite"
        st.session_state.profile_map_type = "satellite"
        st.session_state.selected_profile_name = _DEFAULT_FARM_NAME
        st.session_state.sidebar_profile_choice = _DEFAULT_FARM_NAME
        st.session_state.drive_selected_profile_name = _DEFAULT_FARM_NAME
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


def _percentile(values, pct):
    if not values:
        return 0.0
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (float(pct) / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return data[low]
    weight = rank - low
    return data[low] + (data[high] - data[low]) * weight


def _read_nvidia_gpu_snapshot():
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=1.5).strip()
    except Exception:
        return None
    if not out:
        return None
    first_line = out.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 5:
        return None
    try:
        util_pct = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        power_w = float(parts[4])
    except Exception:
        return None
    mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
    return {
        "name": parts[0] or "NVIDIA GPU",
        "util_pct": max(0.0, min(100.0, util_pct)),
        "mem_used_mb": max(0.0, mem_used),
        "mem_total_mb": max(0.0, mem_total),
        "mem_pct": max(0.0, min(100.0, mem_pct)),
        "power_w": max(0.0, power_w),
        "source": "nvidia-smi",
    }


def _provider_family(provider_name):
    p = str(provider_name or "").lower()
    if "qnn" in p:
        return "npu"
    if "cuda" in p or "dml" in p or "coreml" in p:
        return "gpu"
    if "cpu" in p:
        return "cpu"
    return "other"


def _collect_compute_telemetry():
    now_ts = time.time()
    cached = st.session_state.get("_telemetry_cache")
    if isinstance(cached, dict):
        cached_ts = float(cached.get("ts", 0.0))
        if now_ts - cached_ts <= 0.4 and isinstance(cached.get("data"), dict):
            return dict(cached["data"])

    runtime_info = {}
    if hasattr(engine, "get_runtime_info"):
        try:
            runtime_info = engine.get_runtime_info()
        except Exception:
            runtime_info = {}
    runtime_provider = runtime_info.get("runtime_provider", "Unknown")
    provider_family = _provider_family(runtime_provider)

    if psutil and _PROCESS is not None and "_psutil_primed" not in st.session_state:
        try:
            psutil.cpu_percent(interval=None)
            _PROCESS.cpu_percent(interval=None)
        except Exception:
            pass
        st.session_state._psutil_primed = True

    cpu_total_pct = 0.0
    cpu_process_pct = 0.0
    mem_total_pct = 0.0
    mem_process_mb = 0.0
    if psutil and _PROCESS is not None:
        try:
            cpu_total_pct = float(psutil.cpu_percent(interval=0.0))
            raw_proc_cpu = float(_PROCESS.cpu_percent(interval=0.0))
            cpu_count = max(1, int(psutil.cpu_count(logical=True) or 1))
            cpu_process_pct = max(0.0, min(100.0, raw_proc_cpu / cpu_count))
            mem_total_pct = float(psutil.virtual_memory().percent)
            mem_process_mb = float(_PROCESS.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass

    with _result_lock:
        infer_count = int(_inference_count)
        latest_latency_ms = float(_result_latency_ms)
        last_infer_ts = float(_last_infer_ts)
        latency_values = list(_latency_history_ms)

    if not latency_values and latest_latency_ms > 0:
        latency_values = [latest_latency_ms]

    elapsed_s = max(1e-6, time.time() - float(_inference_started_at))
    infer_fps = infer_count / elapsed_s
    latency_avg_ms = (sum(latency_values) / len(latency_values)) if latency_values else 0.0
    latency_p50_ms = _percentile(latency_values, 50.0)
    latency_p95_ms = _percentile(latency_values, 95.0)

    # How "busy" the active inference accelerator is over time.
    engine_busy_pct = max(0.0, min(100.0, infer_fps * latency_avg_ms / 10.0))

    npu_util_pct = 0.0
    if provider_family == "npu":
        npu_util_pct = engine_busy_pct
    elif any("QNNExecutionProvider" == p for p in runtime_info.get("available_providers", [])):
        npu_util_pct = engine_busy_pct * 0.35

    gpu_snapshot = _read_nvidia_gpu_snapshot()
    gpu_util_pct = 0.0
    gpu_mem_pct = 0.0
    gpu_name = "Unavailable"
    gpu_source = "none"
    gpu_power_w = 0.0
    if gpu_snapshot:
        gpu_util_pct = float(gpu_snapshot["util_pct"])
        gpu_mem_pct = float(gpu_snapshot["mem_pct"])
        gpu_name = str(gpu_snapshot["name"])
        gpu_source = str(gpu_snapshot["source"])
        gpu_power_w = float(gpu_snapshot["power_w"])
    elif provider_family == "gpu":
        gpu_util_pct = engine_busy_pct
        gpu_name = f"{runtime_provider} (estimated)"
        gpu_source = "inference-estimate"

    fps_display = infer_fps if infer_count > 2 else 0.0
    perf_score = 0.0
    if latency_p95_ms > 0:
        perf_score = max(0.0, min(100.0, 100.0 - min(90.0, latency_p95_ms / 2.5)))
    utilization_score = max(0.0, min(100.0, (0.5 * npu_util_pct) + (0.3 * gpu_util_pct) + (0.2 * cpu_process_pct)))
    overall_efficiency = round((0.55 * perf_score) + (0.45 * utilization_score), 1)

    result = {
        "runtime_provider": runtime_provider,
        "provider_family": provider_family,
        "available_providers": runtime_info.get("available_providers", []),
        "cpu_total_pct": round(cpu_total_pct, 1),
        "cpu_process_pct": round(cpu_process_pct, 1),
        "mem_total_pct": round(mem_total_pct, 1),
        "mem_process_mb": round(mem_process_mb, 1),
        "gpu_util_pct": round(gpu_util_pct, 1),
        "gpu_mem_pct": round(gpu_mem_pct, 1),
        "gpu_name": gpu_name,
        "gpu_source": gpu_source,
        "gpu_power_w": round(gpu_power_w, 1),
        "npu_util_pct": round(npu_util_pct, 1),
        "engine_busy_pct": round(engine_busy_pct, 1),
        "inference_count": infer_count,
        "fps": round(fps_display, 2),
        "latency_ms": round(latency_avg_ms, 1),
        "latency_p50_ms": round(latency_p50_ms, 1),
        "latency_p95_ms": round(latency_p95_ms, 1),
        "last_infer_age_s": round(max(0.0, time.time() - last_infer_ts), 1) if last_infer_ts > 0 else None,
        "efficiency_score": overall_efficiency,
    }
    st.session_state["_telemetry_cache"] = {"ts": now_ts, "data": dict(result)}
    return result


def _to_dataframe(rows):
    if pd is None:
        return None
    try:
        return pd.DataFrame(rows)
    except Exception:
        return None


def _render_native_bar(rows, index_col, value_col, label):
    df = _to_dataframe(rows)
    if df is None:
        if rows:
            st.caption(f"{label}: chart fallback table")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No data available for {label.lower()}.")
        return
    if df.empty:
        st.warning(f"No data available for {label.lower()}.")
        return
    if index_col not in df.columns or value_col not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    chart_df = df[[index_col, value_col]].copy().set_index(index_col)
    st.bar_chart(chart_df, use_container_width=True)


@st.cache_data(show_spinner=False, ttl=8)
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

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[: int(max(1, limit))]


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
    map_type="satellite",
    heat_opacity=0.72,
    heat_radius=30,
    marker_threshold=70,
    marker_scale=8.0,
):
    safe_map_type = str(map_type).lower().strip()
    if safe_map_type not in _MAP_TYPES:
        safe_map_type = "satellite"

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

    if str(api_key or "").strip():
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
            + str(api_key) + '&libraries=visualization&callback=initMap"></script>'
            "</head><body><div id=\"map\"></div></body></html>"
        )

    tile_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    tile_attr = "&copy; OpenStreetMap contributors"
    if safe_map_type in {"satellite", "hybrid"}:
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        tile_attr = "Tiles &copy; Esri"
    elif safe_map_type == "terrain":
        tile_url = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        tile_attr = "Map data: &copy; OpenStreetMap contributors, SRTM"

    heat_meters = max(30.0, float(radius) * 18.0)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
        "<style>"
        "*{margin:0;padding:0;box-sizing:border-box}"
        "html,body{background:#08090c}"
        "#map{width:100%;height:" + str(h) + "px;border-radius:10px}"
        ".leaflet-container{background:#08090c;font-family:monospace}"
        ".leaflet-popup-content-wrapper{background:#10121a;color:#eaedf3;border:1px solid #25304b;border-radius:8px}"
        ".leaflet-popup-tip{background:#10121a}"
        "</style></head><body><div id=\"map\"></div>"
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
        "<script>"
        "const DATA=" + points_js + ";"
        "const map=L.map('map',{zoomControl:true,preferCanvas:true});"
        "L.tileLayer('" + tile_url + "',{attribution:'" + tile_attr + "',maxZoom:20}).addTo(map);"
        "if(DATA.length>0){"
        "const b=L.latLngBounds(DATA.map((d)=>[d.lat,d.lng]));"
        "if(b.isValid()){map.fitBounds(b.pad(0.10));}else{map.setView([" + str(clat) + "," + str(clng) + "],16);}"
        "}else{map.setView([" + str(clat) + "," + str(clng) + "],16);}"
        "DATA.forEach((d)=>{"
        "const weight=Math.max(0.05,Math.min(1.0,Number(d.weight||0.05)));"
        "let heatColor='#3a7bff';"
        "if(d.alert>=70){heatColor='#ff3d3d';}else if(d.alert>=40){heatColor='#ffad42';}"
        "const ringOpacity=Math.max(0.08,Math.min(0.65," + str(opacity) + "*weight));"
        "L.circle([d.lat,d.lng],{"
        "radius:Math.max(12,weight*" + str(heat_meters) + "),"
        "stroke:false,fillColor:heatColor,fillOpacity:ringOpacity"
        "}).addTo(map);"
        "if(d.alert>=" + str(threshold) + "){"
        "const m=L.circleMarker([d.lat,d.lng],{"
        "radius:" + str(marker_size) + ",color:'#ffffff',weight:1.5,fillColor:'#ff3d3d',fillOpacity:0.82"
        "}).addTo(map);"
        "m.bindPopup('<b>'+d.label+'</b><br/>Alert: '+Number(d.alert).toFixed(1)+'%<br/>Zone: '+d.zone+' • '+d.risk);"
        "}"
        "});"
        "</script></body></html>"
    )


def _render_data_visualization_page():
    st.markdown("### Farm Disease Heatmap (Sample Dataset)")

    st.markdown("#### Live Recording Dataset")
    live_sessions = _list_live_recording_sessions(limit=80)
    if live_sessions:
        session_ids = [s["session_id"] for s in live_sessions]
        live_selected = st.selectbox(
            "Live session",
            session_ids,
            index=0,
            key="dv_live_session_choice",
        )
        live_meta = next((s for s in live_sessions if s["session_id"] == live_selected), live_sessions[0])
        live_events = _read_live_session_events(live_selected, limit_rows=5000)
        live_detections = [r for r in live_events if str(r.get("event_type", "")).lower() == "detection"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Session", live_selected)
        m2.metric("Farm", str(live_meta.get("farm_name", _DEFAULT_FARM_NAME)))
        m3.metric("Events", int(live_meta.get("events_count", len(live_events))))
        m4.metric("Snapshots", int(live_meta.get("captures_count", 0)))
        session_video_path = _resolve_session_video_path(live_selected, live_meta)
        session_frame_count = int(live_meta.get("video_frame_count", 0) or 0)
        if session_video_path:
            st.caption(
                f"Session video: `{session_video_path}`"
                f" • frames: {session_frame_count}"
                f" • size: {float(live_meta.get('video_size_kb', 0.0)):.1f} KB"
            )
            st.video(session_video_path)
        else:
            st.caption("No session video file found yet for this session.")

        if live_detections:
            live_points = []
            for r in live_detections:
                try:
                    lat = float(r.get("latitude", st.session_state.active_farm_lat))
                    lng = float(r.get("longitude", st.session_state.active_farm_lng))
                    conf_pct = float(r.get("confidence_pct", 0.0))
                except Exception:
                    continue
                live_points.append(
                    {
                        "lat": lat,
                        "lng": lng,
                        "disease_label": str(r.get("disease_label", "Detection")),
                        "disease_alert_pct": max(0.0, min(100.0, conf_pct)),
                        "risk_band": "High" if conf_pct >= 70 else ("Moderate" if conf_pct >= 40 else "Low"),
                        "zone": str(r.get("coord_source", "session")),
                    }
                )
            if live_points:
                center_lat_live = sum(p["lat"] for p in live_points) / max(1, len(live_points))
                center_lng_live = sum(p["lng"] for p in live_points) / max(1, len(live_points))
                live_map_html = _build_analytics_heatmap_html(
                    api_key=GOOGLE_MAPS_API_KEY,
                    center_lat=float(center_lat_live),
                    center_lng=float(center_lng_live),
                    points=live_points,
                    map_height=420,
                    map_type=_coerce_map_type(st.session_state.map_base_type),
                    heat_opacity=0.75,
                    heat_radius=30,
                    marker_threshold=55.0,
                    marker_scale=float(st.session_state.disease_spot_size),
                )
                components.html(live_map_html, height=440, scrolling=False)

            st.dataframe(
                [
                    {
                        "Time": r.get("time", ""),
                        "Label": r.get("disease_label", ""),
                        "Confidence %": r.get("confidence_pct", ""),
                        "Lat": r.get("latitude", ""),
                        "Lng": r.get("longitude", ""),
                        "Source": r.get("coord_source", ""),
                        "Map": r.get("map_layer", ""),
                        "Scheme": r.get("color_scheme", ""),
                        "Snapshot": r.get("snapshot_path", ""),
                    }
                    for r in live_detections[:80]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No detection events logged for the selected live session yet.")
    else:
        st.caption("No live recording sessions found yet. Start the camera to create one.")

    st.markdown("#### Farm/Date Detection Tables")
    detection_files = _list_detection_dataset_files(limit=200)
    if detection_files:
        farm_options = sorted({f["farm_slug"] for f in detection_files})
        selected_farm_slug = st.selectbox("Detection farm", farm_options, index=0, key="dv_det_farm_choice")
        farm_rows = [f for f in detection_files if f["farm_slug"] == selected_farm_slug]
        date_options = [f["date"] for f in farm_rows]
        selected_date = st.selectbox("Detection date", date_options, index=0, key="dv_det_date_choice")
        selected_file = next((f for f in farm_rows if f["date"] == selected_date), farm_rows[0])
        det_rows = _read_detection_dataset_rows(selected_file["path"], limit_rows=3000)
        st.caption(f"Dataset file: `{selected_file['path']}`")
        if det_rows:
            st.dataframe(
                [
                    {
                        "Timestamp": r.get("timestamp", ""),
                        "Farm": r.get("farm_name", ""),
                        "Label": r.get("disease_label", ""),
                        "Confidence %": r.get("confidence_pct", ""),
                        "Lat": r.get("latitude", ""),
                        "Lng": r.get("longitude", ""),
                        "Source": r.get("coord_source", ""),
                    }
                    for r in det_rows[:250]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No rows found in the selected detection dataset.")
    else:
        st.caption("No farm/date detection datasets found yet.")

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
    if not dataset:
        st.warning("Sample dataset is empty. Adjust sample controls and try again.")
        return

    st.caption(
        f"Using generated sample dataset for `{st.session_state.active_farm_name}`"
        f" • points: {len(dataset)} • seed: {int(sample_seed)}"
    )

    st.markdown("#### Map Heat Layer (Sample Dataset)")
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

    preview_rows = dataset[: min(10, len(dataset))]
    dataset_df = _to_dataframe(dataset)
    st.markdown("#### Sample Spatial Data And Rows")
    p1, p2 = st.columns([1.5, 1.1])
    with p1:
        st.markdown("#### Sample Spatial Plot")
        if dataset_df is not None and not dataset_df.empty:
            st.scatter_chart(
                dataset_df,
                x="lng",
                y="lat",
                size="disease_alert_pct",
                use_container_width=True,
            )
        else:
            st.warning("Could not render scatter chart; showing sample rows instead.")
            st.dataframe(
                [
                    {
                        "Disease": r["disease_label"],
                        "Alert %": r["disease_alert_pct"],
                        "Zone": r["zone"],
                        "Lat": r["lat"],
                        "Lng": r["lng"],
                    }
                    for r in preview_rows
                ],
                use_container_width=True,
                hide_index=True,
                height=330,
            )
    with p2:
        st.markdown("#### Sample Rows")
        st.dataframe(
            [
                {
                    "Disease": r["disease_label"],
                    "Alert %": r["disease_alert_pct"],
                    "Zone": r["zone"],
                    "Lat": r["lat"],
                    "Lng": r["lng"],
                }
                for r in preview_rows
            ],
            use_container_width=True,
            hide_index=True,
            height=330,
        )

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
        total_counts = max(1, sum(disease_counts.values()))
        share_rows = []
        for row in dist_rows:
            share_rows.append(
                {
                    "Disease": row["Disease"],
                    "SharePct": round((float(row["Count"]) / total_counts) * 100.0, 2),
                }
            )
        _render_native_bar(share_rows, "Disease", "SharePct", "Disease share")
    elif dist_mode == "Average alert by disease":
        _render_native_bar(dist_rows, "Disease", "AvgAlert", "Average alert")
    else:
        _render_native_bar(dist_rows, "Disease", "Count", "Disease counts")

    st.markdown("#### Additional Analytics Graphs")
    extra_mode = st.selectbox(
        "Additional graph",
        ["Risk bands", "Environmental averages", "Alert histogram", "Top zones by detections"],
        key="dv_extra_mode",
    )

    if extra_mode == "Risk bands":
        risk_rows = [{"Band": k, "Count": v} for k, v in sorted(risk_counts.items())]
        _render_native_bar(risk_rows, "Band", "Count", "Risk bands")
    elif extra_mode == "Environmental averages":
        n = max(1, len(dataset))
        env_rows = [
            {"Metric": "Moisture %", "Value": round(env_totals["moisture_pct"] / n, 2)},
            {"Metric": "Humidity %", "Value": round(env_totals["humidity_pct"] / n, 2)},
            {"Metric": "Temperature C", "Value": round(env_totals["temp_c"] / n, 2)},
            {"Metric": "Crop Stress %", "Value": round(env_totals["crop_stress_pct"] / n, 2)},
        ]
        _render_native_bar(env_rows, "Metric", "Value", "Environmental averages")
    elif extra_mode == "Alert histogram":
        hist_rows = []
        for start in range(0, 100, 10):
            end = start + 10
            count_bin = sum(1 for a in alerts if (a >= start and (a < end or end == 100)))
            hist_rows.append({"Range": f"{start}-{end}", "Count": count_bin})
        _render_native_bar(hist_rows, "Range", "Count", "Alert histogram")
    else:
        top_zones = sorted(zone_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        zone_rows = [{"Zone": z, "Count": c} for z, c in top_zones]
        _render_native_bar(zone_rows, "Zone", "Count", "Top zones")

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
    st.markdown("### On-Device Utilization, Latency, and Performance")
    refresh_left, refresh_right = st.columns([1.3, 2.2])
    refresh_left.toggle("Auto-refresh (1s)", key="compute_auto_refresh")
    if st.session_state.compute_auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=1000, key="compute_1s_autorefresh")
            refresh_right.caption("Live telemetry refresh is active (1 second).")
        else:
            refresh_right.warning("Install `streamlit-autorefresh` to enable 1-second updates.")

    if psutil is None:
        st.warning("`psutil` is not installed in this environment. CPU and memory telemetry will be limited.")
    telemetry = _collect_compute_telemetry()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPU (System)", f"{telemetry['cpu_total_pct']:.1f}%")
    c2.metric("GPU Utilization", f"{telemetry['gpu_util_pct']:.1f}%")
    c3.metric("NPU Utilization", f"{telemetry['npu_util_pct']:.1f}%")
    c4.metric("Avg Latency", f"{telemetry['latency_ms']:.1f} ms")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("P95 Latency", f"{telemetry['latency_p95_ms']:.1f} ms")
    c6.metric("Inference FPS", f"{telemetry['fps']:.2f}")
    c7.metric("Inferences", str(telemetry["inference_count"]))
    c8.metric("Efficiency Score", f"{telemetry['efficiency_score']:.1f}/100")

    if telemetry["gpu_source"] == "none":
        st.caption("GPU telemetry source was not detected on this machine. GPU utilization may appear as 0% unless a GPU runtime provider is active.")

    st.progress(telemetry["cpu_total_pct"] / 100.0, text=f"CPU load {telemetry['cpu_total_pct']:.1f}%")
    st.progress(telemetry["gpu_util_pct"] / 100.0, text=f"GPU load {telemetry['gpu_util_pct']:.1f}%")
    st.progress(telemetry["npu_util_pct"] / 100.0, text=f"NPU load {telemetry['npu_util_pct']:.1f}%")

    util_rows = [
        {"Resource": "CPU (System)", "Utilization": telemetry["cpu_total_pct"]},
        {"Resource": "CPU (App)", "Utilization": telemetry["cpu_process_pct"]},
        {"Resource": "GPU", "Utilization": telemetry["gpu_util_pct"]},
        {"Resource": "NPU", "Utilization": telemetry["npu_util_pct"]},
        {"Resource": "Inference Duty", "Utilization": telemetry["engine_busy_pct"]},
    ]
    _render_native_bar(util_rows, "Resource", "Utilization", "Utilization")

    runtime_provider = telemetry["runtime_provider"]
    provider_family = telemetry["provider_family"]
    gpu_note = f"{telemetry['gpu_name']} • source: {telemetry['gpu_source']}"
    if telemetry["gpu_power_w"] > 0:
        gpu_note += f" • power: {telemetry['gpu_power_w']:.1f} W"

    with _result_lock:
        latency_values = list(_latency_history_ms)
    if latency_values:
        latency_bins = []
        bucket = 5
        for start in range(0, 205, bucket):
            end = start + bucket
            count_bin = sum(1 for val in latency_values if start <= val < end)
            if count_bin > 0:
                latency_bins.append({"Range": f"{start}-{end} ms", "Count": count_bin})
        if latency_bins:
            st.markdown("#### Latency Distribution")
            _render_native_bar(latency_bins, "Range", "Count", "Latency distribution")

    st.markdown("#### Runtime Details")
    st.dataframe(
        [
            {"Metric": "Runtime provider", "Value": runtime_provider},
            {"Metric": "Provider family", "Value": provider_family.upper()},
            {"Metric": "Available providers", "Value": ", ".join(telemetry["available_providers"]) or "Unknown"},
            {"Metric": "GPU details", "Value": gpu_note},
            {"Metric": "System memory", "Value": f"{telemetry['mem_total_pct']:.1f}%"},
            {"Metric": "App memory (RSS)", "Value": f"{telemetry['mem_process_mb']:.1f} MB"},
            {"Metric": "Last inference age", "Value": "n/a" if telemetry["last_infer_age_s"] is None else f"{telemetry['last_infer_age_s']:.1f} s"},
            {"Metric": "Host platform", "Value": f"{platform.system()} {platform.release()}"},
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Rubric Snapshot")
    st.dataframe(
        [
            {
                "Evaluation Area": "Resource utilization",
                "Current Signal": f"CPU {telemetry['cpu_total_pct']:.1f}% • GPU {telemetry['gpu_util_pct']:.1f}% • NPU {telemetry['npu_util_pct']:.1f}%",
            },
            {
                "Evaluation Area": "Latency",
                "Current Signal": f"Avg {telemetry['latency_ms']:.1f} ms • P95 {telemetry['latency_p95_ms']:.1f} ms",
            },
            {
                "Evaluation Area": "Performance",
                "Current Signal": f"{telemetry['fps']:.2f} FPS • {telemetry['inference_count']} total inferences",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_gallery_page():
    st.markdown("### Recordings and Images")
    live_sessions = _list_live_recording_sessions(limit=120)
    session_ids = [s["session_id"] for s in live_sessions]

    f1, f2, f3, f4 = st.columns([1.0, 1.2, 1.2, 1.0])
    media_type = f1.selectbox("Type", ["all", "image", "video"], index=0, key="gallery_type_filter")
    source = f2.selectbox(
        "Source",
        ["all", "recordings/captures", "live sessions", "dataset"],
        index=0,
        key="gallery_source_filter",
    )
    session_filter_options = ["all"] + session_ids
    selected_session_filter = f3.selectbox(
        "Session filter",
        session_filter_options,
        index=0,
        key="gallery_session_filter",
    )
    st.session_state.gallery_limit = int(
        f4.slider("Max items", min_value=12, max_value=120, value=int(st.session_state.gallery_limit), step=12, key="gallery_limit_slider")
    )

    items = _scan_gallery_media(limit=max(120, int(st.session_state.gallery_limit)))
    if source != "all":
        if source == "dataset":
            items = [x for x in items if x["path"].startswith("dataset" + os.sep) or x["path"].startswith("Agri-Scout" + os.sep + "dataset")]
        elif source == "live sessions":
            items = [
                x for x in items
                if ("dataset" + os.sep + "live_sessions" + os.sep) in x["path"]
                or ("captures" + os.sep + "live_sessions" + os.sep) in x["path"]
                or ("recordings" + os.sep + "live_sessions" + os.sep) in x["path"]
            ]
        else:
            items = [x for x in items if x["path"].startswith("recordings" + os.sep) or x["path"].startswith("captures" + os.sep)]
    if selected_session_filter != "all":
        token_a = os.sep + selected_session_filter + os.sep
        token_b = selected_session_filter + os.sep
        items = [x for x in items if (token_a in x["path"] or token_b in x["path"])]
    if media_type != "all":
        items = [x for x in items if x["kind"] == media_type]

    items = items[: int(st.session_state.gallery_limit)]
    st.caption(f"Showing {len(items)} items")
    if not items:
        st.info("No media matched your filters.")
    else:
        for item in items:
            expander_label = f"{item['name']}  •  {item['kind']}  •  {item['size_kb']} KB  •  {item['parent']}"
            with st.expander(expander_label, expanded=False):
                st.caption(f"Path: {item['path']}")
                try:
                    if item["kind"] == "image":
                        st.image(item["path"], use_container_width=True)
                    else:
                        st.video(item["path"])
                except Exception as e:
                    st.warning(f"Could not render `{item['path']}`: {e}")

    st.markdown("#### Session Event Logs")
    if not live_sessions:
        st.caption("No live recording sessions found.")
        return

    log_sid = st.selectbox("Session data", session_ids, index=0, key="gallery_session_data_choice")
    log_meta = next((s for s in live_sessions if s["session_id"] == log_sid), live_sessions[0])
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Session", log_sid)
    g2.metric("Events", int(log_meta.get("events_count", 0)))
    g3.metric("Snapshots", int(log_meta.get("captures_count", 0)))
    g4.metric("Video frames", int(log_meta.get("video_frame_count", 0)))
    gallery_video_path = _resolve_session_video_path(log_sid, log_meta)
    if gallery_video_path:
        st.caption(
            f"Session video: `{gallery_video_path}`"
            f" • size: {float(log_meta.get('video_size_kb', 0.0)):.1f} KB"
        )
        st.video(gallery_video_path)
    else:
        st.caption("No session video found for this session.")

    log_rows = _read_live_session_events(log_sid, limit_rows=3000)
    if not log_rows:
        st.caption("No events found for this session.")
        return

    event_type_filter = st.selectbox(
        "Event type",
        ["all", "inference", "detection"],
        index=0,
        key="gallery_event_type_filter",
    )
    if event_type_filter != "all":
        log_rows = [r for r in log_rows if str(r.get("event_type", "")).lower() == event_type_filter]

    st.dataframe(
        [
            {
                "Timestamp": r.get("timestamp", ""),
                "Type": r.get("event_type", ""),
                "Label": r.get("disease_label", ""),
                "Confidence %": r.get("confidence_pct", ""),
                "Lat": r.get("latitude", ""),
                "Lng": r.get("longitude", ""),
                "Source": r.get("coord_source", ""),
                "Map": r.get("map_layer", ""),
                "Scheme": r.get("color_scheme", ""),
                "Snapshot": r.get("snapshot_path", ""),
            }
            for r in log_rows[:500]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_settings_page():
    st.markdown("### Settings")
    if st.session_state.settings_notice:
        st.success(st.session_state.settings_notice)
        st.session_state.settings_notice = ""

    st.markdown("#### Display")

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
        st.session_state.pending_apply_farm = True
        st.rerun()
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
        st.session_state.pending_save_profile_name = str(st.session_state.profile_name_input)
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


def _render_help_page():
    st.markdown("### Help")
    st.markdown(
        "1. Start live camera feed with **START**.\n"
        "2. Keep canopy centered and maintain stable movement.\n"
        "3. Use the **Settings** tab for color schemes and farm location.\n"
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


def _render_app_title():
    if _APP_LOGO_PATH.exists():
        st.image(str(_APP_LOGO_PATH), width=92)
    else:
        st.markdown(
            '<div class="app-title-wrap">'
            '<div class="app-title-main">TRACTION</div>'
            '<div class="app-title-sub">Real-Time Crop Disease Detection And Field Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

_apply_pending_actions()

with st.sidebar:

    st.markdown(
        '<div style="text-align:center;padding:0.6rem 0 0.2rem 0">'
        '<span style="font-size:1.4rem">🌱</span>'
        '<span style="font-size:0.9rem;font-weight:700;color:#eaedf3;'
        'margin-left:8px;letter-spacing:-0.3px">TRACTION</span>'
        '<span style="font-size:0.55rem;color:#3d4250;margin-left:6px;'
        "font-family:'IBM Plex Mono',monospace\">v4.0</span>"
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Quick Controls</div>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    q1.toggle("Tractor mode", value=st.session_state.tractor_mode, key="tractor_mode")
    q2.toggle("Sunlight boost", value=st.session_state.sunlight_mode, key="sunlight_mode")
    st.toggle(
        "Advanced telemetry",
        value=st.session_state.show_advanced_ui,
        key="show_advanced_ui",
    )

    if st.button("🔄 Refresh Now", use_container_width=True, type="primary", key="sb_refresh_now"):
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Field Quick Access</div>', unsafe_allow_html=True)
    profile_names = [p.get("name", "Unnamed") for p in st.session_state.map_profiles]
    if profile_names:
        if st.session_state.sidebar_profile_choice not in profile_names:
            st.session_state.sidebar_profile_choice = profile_names[0]
        st.selectbox("Saved profiles", profile_names, key="sidebar_profile_choice")
        ap1, ap2 = st.columns(2)
        if ap1.button("Load Profile", use_container_width=True, key="sb_load_profile"):
            st.session_state.pending_load_profile_name = st.session_state.sidebar_profile_choice
            st.rerun()
        if ap2.button("Reset Farm", use_container_width=True, key="sb_reset_profile"):
            st.session_state.pending_reset_farm = True
            st.rerun()

    c1, c2 = st.columns(2)
    c1.number_input(
        "Lat",
        min_value=-90.0,
        max_value=90.0,
        step=0.0001,
        format="%.6f",
        key="sidebar_quick_lat",
    )
    c2.number_input(
        "Lng",
        min_value=-180.0,
        max_value=180.0,
        step=0.0001,
        format="%.6f",
        key="sidebar_quick_lng",
    )
    if st.button("Apply Coordinates", use_container_width=True, key="sb_apply_coords"):
        st.session_state.drive_lat_input = float(st.session_state.sidebar_quick_lat)
        st.session_state.drive_lng_input = float(st.session_state.sidebar_quick_lng)
        st.session_state.pending_drive_coord_apply = True
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Farm Snapshot</div>', unsafe_allow_html=True)
    speed_val = sim.get_speed_mph() if sim else 5.0
    area_acres = (float(st.session_state.active_farm_width_m) * float(st.session_state.active_farm_height_m)) / 4046.8564224
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Active Farm</div>'
        '<div class="tc-val" style="font-size:0.95rem">' + farm_name_display + '</div>'
        '<div class="tc-sub">' + f"{st.session_state.active_farm_lat:.5f}, {st.session_state.active_farm_lng:.5f}" + "</div>"
        '<div class="tc-sub">' + f"{area_acres:.2f} acres" + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Drive Status</div>'
        '<div class="tc-val">' + f"{speed_val:.1f}" + ' <span class="tc-unit">MPH</span>  •  '
        + f"{tractor_heading:.0f}° {heading_cardinal}" + '</div>'
        '<div class="tc-sub">Last detection: ' + last_det_age_text + '  •  Distance: ' + last_det_dist_text + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Detections Logged</div>'
        '<div class="tc-val">' + red_dot + str(disease_count) + '</div>'
        '<div class="tc-sub">Geo-tagged events with debounce filtering</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Live Recording</div>', unsafe_allow_html=True)
    active_sid = str(st.session_state.active_recording_id)
    if active_sid:
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Active Session</div>'
            '<div class="tc-val" style="font-size:0.78rem">' + active_sid + '</div>'
            '<div class="tc-sub">Events: ' + str(int(st.session_state.active_recording_event_count))
            + '  •  Snapshots: ' + str(int(st.session_state.active_recording_capture_count))
            + '  •  Frames: ' + str(int(st.session_state.active_recording_frame_count)) + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tcard">'
            '<div class="tc-label">Active Session</div>'
            '<div class="tc-val" style="font-size:0.82rem">none</div>'
            '<div class="tc-sub">Press START on camera to begin a new dataset session.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    sessions_for_sidebar = _list_live_recording_sessions(limit=40)
    if sessions_for_sidebar:
        session_ids_sidebar = [s["session_id"] for s in sessions_for_sidebar]
        if st.session_state.sidebar_session_choice not in session_ids_sidebar:
            st.session_state.sidebar_session_choice = session_ids_sidebar[0]
        st.selectbox("Recent sessions", session_ids_sidebar, key="sidebar_session_choice")

    telemetry = _collect_compute_telemetry()
    st.markdown('<div class="sidebar-section-label">Compute Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Runtime Provider</div>'
        '<div class="tc-val" style="font-size:0.85rem">' + telemetry["runtime_provider"] + '</div>'
        '<div class="tc-sub">FPS: ' + f"{telemetry['fps']:.2f}" + '  •  Avg latency: '
        + f"{telemetry['latency_ms']:.1f} ms" + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Utilization</div>'
        '<div class="tc-val" style="font-size:0.82rem">CPU '
        + f"{telemetry['cpu_total_pct']:.1f}%  •  GPU {telemetry['gpu_util_pct']:.1f}%  •  NPU {telemetry['npu_util_pct']:.1f}%"
        + '</div>'
        '<div class="tc-sub">Efficiency score: ' + f"{telemetry['efficiency_score']:.1f}/100" + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    session_time = datetime.datetime.now().strftime("%b %d, %Y — %H:%M")
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Session Time</div>'
        '<div class="tc-val" style="font-size:0.82rem">' + session_time + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

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
        .tcard,.feed-wrap,.det-panel,.advisory-panel,.map-panel,.quick-card,.cab-alert,.mcard{
          border-width:3px !important;
          box-shadow:0 0 0 1px rgba(255,255,255,0.08),0 8px 20px rgba(0,0,0,0.35) !important;
        }
        .det-class,.cab-alert-state,.quick-val,.tc-val{
          font-size:1.28rem !important;
          font-weight:800 !important;
          letter-spacing:0.2px !important;
        }
        .feed-label,.tc-label,.quick-label,.map-stat,.adv-status{
          color:var(--ag-text) !important;
          font-weight:700 !important;
        }
        .stButton button,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        button[role="switch"]{
          border-width:3px !important;
          font-weight:700 !important;
        }
        section[data-testid="stSidebar"] .stButton button{
          min-height:62px !important;
          font-size:1.05rem !important;
        }
        .stTabs [data-baseweb="tab"]{
          font-weight:700 !important;
        }
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
        camera_playing = bool(ctx.state.playing)
        _sync_live_recording_state(camera_playing)
        coord_source = "tractor_sim" if st.session_state.tractor_sim is not None else "placeholder"
        sample_lat = tractor_lat
        sample_lng = tractor_lng
        if st.session_state.tractor_sim is None:
            sample_lat, sample_lng = _build_placeholder_detection_coords()
        if camera_playing:
            _log_live_inference_sample(sample_lat, sample_lng, coord_source)
        if st.session_state.recording_notice:
            st.caption(st.session_state.recording_notice)
            st.session_state.recording_notice = ""
        if st.session_state.tractor_mode and not ctx.state.playing:
            st.caption("Tap START, then allow camera permission to begin live scanning.")

        active_session = str(st.session_state.active_recording_id)
        rec_cols = st.columns(5)
        rec_cols[0].metric("Recording", "Active" if active_session else "Idle")
        rec_cols[1].metric("Session", active_session or "none")
        rec_cols[2].metric("Events", int(st.session_state.active_recording_event_count))
        rec_cols[3].metric("Snapshots", int(st.session_state.active_recording_capture_count))
        rec_cols[4].metric("Video Frames", int(st.session_state.active_recording_frame_count))

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
    st.markdown("#### AI Treatment Advisory")
    advisory_text = st.session_state.advisory_text
    if advisory_text:
        st.success(advisory_text)
    else:
        if is_waiting or is_healthy:
            st.info("Awaiting disease detection. Advisory appears after a confirmed detection event.")
        else:
            st.warning(f"Disease detected: {label} at {conf_pct}%. Advisory is being generated.")

    recent_events = _read_live_session_events(st.session_state.active_recording_id, limit_rows=40) if st.session_state.active_recording_id else []
    if recent_events:
        st.dataframe(
            [
                {
                    "Time": r.get("time", ""),
                    "Type": r.get("event_type", ""),
                    "Label": r.get("disease_label", ""),
                    "Confidence %": r.get("confidence_pct", ""),
                    "Source": r.get("coord_source", ""),
                    "Map": r.get("map_layer", ""),
                    "Scheme": r.get("color_scheme", ""),
                    "Snapshot": r.get("snapshot_path", ""),
                }
                for r in recent_events[:12]
            ],
            use_container_width=True,
            hide_index=True,
        )

    profile_names = [p.get("name", "Unnamed") for p in st.session_state.map_profiles]
    if profile_names and st.session_state.drive_selected_profile_name not in profile_names:
        st.session_state.drive_selected_profile_name = profile_names[0]

    map_ctrl_left, map_ctrl_mid, map_ctrl_profile, map_ctrl_apply = st.columns([2.1, 2.0, 2.6, 1.3])
    with map_ctrl_left:
        st.toggle("Precision map pan/zoom", value=st.session_state.map_interactive, key="map_interactive")
    with map_ctrl_mid:
        st.selectbox(
            "Map layer",
            _MAP_TYPES,
            index=_MAP_TYPES.index(_coerce_map_type(st.session_state.map_base_type)),
            key="map_base_type",
        )
    with map_ctrl_profile:
        if profile_names:
            st.selectbox("Saved farms", profile_names, key="drive_selected_profile_name")
        else:
            st.caption("No saved farms available.")
    with map_ctrl_apply:
        st.markdown('<div style="height:1.62rem"></div>', unsafe_allow_html=True)
        if st.button("Switch Farm", use_container_width=True, key="drive_switch_farm_btn", disabled=(not profile_names)):
            st.session_state.pending_load_profile_name = st.session_state.drive_selected_profile_name
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



_render_app_title()

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
