#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Real-Time Crop Disease Detection Dashboard  (v4 — Ghost Tractor)
================================================================================
 Run with:   streamlit run app.py
================================================================================
"""

import datetime
import math
import random
import threading
import time
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
section[data-testid="stSidebar"]{background:#0c0e14;border-right:1px solid #1a1d28}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#a0a4b0}
.hdr{background:linear-gradient(135deg,#0d1117,#111820);border:1px solid #1a1f2e;border-radius:14px;padding:1rem 1.6rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between}
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
.tcard .tc-val{font-size:1.15rem;font-weight:600;color:#e0e3eb;font-family:'IBM Plex Mono',monospace;line-height:1.3}
.tcard .tc-unit{font-size:0.7rem;color:#3d9970;font-weight:500}
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
.adv-status{font-size:0.6rem;color:#3d4250;text-transform:uppercase;letter-spacing:1px;font-family:'IBM Plex Mono',monospace;font-weight:500}
.adv-body{background:#0a0c12;border:1px solid #141722;border-radius:10px;padding:1rem 1.1rem;min-height:100px}
.adv-placeholder{color:#2e3240;font-size:0.82rem;font-family:'IBM Plex Mono',monospace;line-height:1.7}
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
if "tractor_sim" not in st.session_state:
    if _MAP_AVAILABLE:
        st.session_state.tractor_sim = TractorSimulator()
    else:
        st.session_state.tractor_sim = None


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


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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
        '<div class="tc-label">Engine</div>'
        '<div class="tc-val">1,850 <span class="tc-unit">RPM</span></div>'
        '<div class="tc-sub">John Deere 8R — Cruise ON</div>'
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

    # ---- Navigation ----
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

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
        '<div class="tc-sub">Iowa County, WI  •  Simulated</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Compute ----
    st.markdown('<div class="sidebar-section-label">Compute</div>', unsafe_allow_html=True)

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

    # ---- Field Intel ----
    st.markdown('<div class="sidebar-section-label">Field Intel</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Disease Detections Logged</div>'
        '<div class="tc-val">' + red_dot + str(disease_count) + '</div>'
        '<div class="tc-sub">Geo-tagged  •  5s debounce</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tcard">'
        '<div class="tc-label">Active Field</div>'
        '<div class="tc-val" style="font-size:0.9rem">North-40</div>'
        '<div class="tc-sub">Corn (V8-VT)  •  140 acres</div>'
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
    st.caption("Live auto-refresh is disabled to prevent map flicker. Use the refresh button below.")
    st.session_state.map_interactive = st.toggle(
        "Enable map pan/zoom (precision mode)",
        value=st.session_state.map_interactive,
    )
    st.session_state.map_zoom_level = st.slider(
        "Map zoom level",
        min_value=16,
        max_value=22,
        value=int(st.session_state.map_zoom_level),
        step=1,
    )

    # ---- Refresh button (updates map + telemetry) ----
    if st.button("🔄 Refresh Map & Telemetry", use_container_width=True):
        st.rerun()

    st.markdown(
        '<div style="text-align:center;font-size:0.55rem;color:#2a2d38;'
        "font-family:'IBM Plex Mono',monospace;letter-spacing:0.8px;margin-top:8px\">"
        'QUALCOMM EDGE AI HACKATHON 2026</div>',
        unsafe_allow_html=True,
    )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN CONTENT                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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

col_feed, col_status = st.columns([5, 2])

with col_feed:
    st.markdown(
        '<div class="feed-label"><span class="live-dot green"></span> Live Camera Feed</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="feed-wrap">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="agri-scout-cam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col_status:
    st.markdown('<div class="feed-label">Detection Status</div>', unsafe_allow_html=True)

    label = st.session_state.last_class
    conf = st.session_state.last_conf
    count = st.session_state.frame_count

    is_healthy = "healthy" in label.lower()
    is_waiting = "waiting" in label.lower() or "initial" in label.lower()

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


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TREATMENT ADVISORY                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

advisory_text = st.session_state.advisory_text

st.markdown('<div class="advisory-panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="adv-header">'
    '<div class="adv-header-left">'
    '<div class="adv-icon">🧬</div>'
    '<span class="adv-title">AI Treatment Advisory</span></div>'
    '<span class="adv-status">Powered by LLM Agent</span></div>',
    unsafe_allow_html=True,
)

if advisory_text:
    st.markdown(
        '<div class="adv-body"><div class="adv-content">' + advisory_text + '</div></div>',
        unsafe_allow_html=True,
    )
else:
    if is_waiting or is_healthy:
        placeholder = (
            '<span style="color:#2a3040">$</span> awaiting disease detection…'
            '<span class="cursor"></span><br>'
            '<span style="color:#1e2230">// Treatment plan will appear when a disease is detected.</span>'
        )
    else:
        placeholder = (
            '<span style="color:#ff6b6b">⚡</span> '
            '<span style="color:#5a6072">Disease detected:</span> '
            '<span style="color:#eaedf3;font-weight:600">' + label + '</span> '
            '<span style="color:#5a6072">at</span> '
            '<span style="color:#eaedf3">' + conf_pct + '%</span><br>'
            '<span style="color:#2a3040">$</span> generating treatment advisory…'
            '<span class="cursor"></span>'
        )
    st.markdown(
        '<div class="adv-body"><div class="adv-placeholder">' + placeholder + '</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="adv-footer">'
    '<div class="adv-footer-item">Model: Claude / Llama 3</div>'
    '<div class="adv-footer-item">Context: Field conditions + USDA guidelines</div>'
    '<div class="adv-footer-item">Refresh: On detection change</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  GHOST TRACTOR MAP                                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _render_map_panel(map_lat, map_lng, map_heading, map_disease_count):
    sim_state = st.session_state.tractor_sim
    map_trail_count = len(sim_state.path_history) if sim_state is not None else 0
    map_height = 420

    st.markdown('<div class="map-panel">', unsafe_allow_html=True)

    if _MAP_AVAILABLE and sim_state is not None:
        try:
            st.markdown(
                '<div class="map-header">'
                '<div class="map-header-left">'
                '<div class="map-icon">🗺️</div>'
                '<span class="map-title">Ghost Tractor — Field Map</span></div>'
                '<div class="map-stats">'
                '<span class="map-stat">Detections: <strong>' + str(map_disease_count) + '</strong></span>'
                '<span class="map-stat">Trail: ' + str(map_trail_count) + ' pts</span>'
                '<span class="map-stat">View: ' + ('Interactive' if st.session_state.map_interactive else 'Locked') + ' Satellite</span>'
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
            )
            components.html(map_html, height=map_height, scrolling=False)
        except Exception as e:
            st.error(f"Map rendering error: {e}")
    else:
        st.markdown(
            '<div class="map-header">'
            '<div class="map-header-left">'
            '<div class="map-icon">🗺️</div>'
            '<span class="map-title">Ghost Tractor — Field Map</span></div>'
            '<div class="map-stats">'
            '<span class="map-stat">Detections: <strong>' + str(map_disease_count) + '</strong></span>'
            '<span class="map-stat">Trail: ' + str(map_trail_count) + ' pts</span>'
            '<span class="map-stat">View: ' + ('Interactive' if st.session_state.map_interactive else 'Locked') + ' Satellite</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "**Map unavailable.** Make sure `map_engine.py` is in the same "
            "directory as `app.py` and restart Streamlit."
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_map_panel_static():
    _render_map_panel(
        tractor_lat,
        tractor_lng,
        tractor_heading,
        len(st.session_state.disease_logs),
    )


render_map_panel_static()


# ---- Bottom metrics ----
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
# --- HEATMAP STRESS TESTER ---
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
