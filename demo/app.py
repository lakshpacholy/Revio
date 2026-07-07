"""
Revio — Streamlit Demo UI

Full offline pipeline:
  Audio clip -> MFCC extraction -> TFLite CNN -> Risk Scorer -> Narrative -> TTS
"""

import sys
import time
from pathlib import Path

# Ensure repo root is on path when launched from demo/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from src.preprocess import load_audio, extract_mfcc, DURATION
from src.edge_inference import TFLiteClassifier
from src.risk_scorer import score as risk_score, TIER_CRITICAL, TIER_WARNING
from src.narrative import get_narrative
from src.tts_alert import alert as tts_alert

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Revio — Vehicle Health Monitor",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR     = ROOT / "models"
DEMO_CLIPS_DIR = Path(__file__).parent / "demo_clips"

STATE_CONFIG = {
    "Braking": {
        "model":   MODELS_DIR / "braking_model.tflite",
        "labels":  ["normal_brakes", "worn_out_brakes"],
        "description": "Detects worn brake pads from braking sounds",
    },
    "Start-Up": {
        "model":   MODELS_DIR / "startup_model.tflite",
        "labels":  ["normal_engine_startup", "bad_ignition", "dead_battery"],
        "description": "Detects ignition faults and battery issues at startup",
    },
    "Idle": {
        "model":   MODELS_DIR / "idle_model.tflite",
        "labels":  ["normal_engine_idle", "low_oil", "power_steering", "serpentine_belt"],
        "description": "Detects oil, power steering, and belt faults at idle",
    },
}

# Demo clip metadata: filename prefix -> (display name, recommended state)
DEMO_CLIPS = {
    "braking_normal_brakes_1.wav":  ("Normal Brakes",        "Braking"),
    "braking_worn_out_brakes_1.wav":("Worn-Out Brakes",      "Braking"),
    "startup_normal_1.wav":         ("Normal Startup",        "Start-Up"),
    "startup_bad_ignition_1.wav":   ("Bad Ignition",          "Start-Up"),
    "startup_dead_battery_1.wav":   ("Dead Battery",          "Start-Up"),
    "idle_normal_1.wav":            ("Normal Idle",           "Idle"),
    "idle_low_oil_1.wav":           ("Low Oil",               "Idle"),
    "idle_power_steering_1.wav":    ("Power Steering Fault",  "Idle"),
    "idle_serpentine_belt_1.wav":   ("Serpentine Belt Fault", "Idle"),
}

# Cosmetic colour tokens — visual only, mirrors the palette used across the UI.
TIER_COLOURS = {
    "CRITICAL": "#EF4444",
    "WARNING":  "#F59E0B",
    "MONITOR":  "#3B82F6",
    "NORMAL":   "#10B981",
}

TIER_EMOJI = {
    "CRITICAL": "🔴",
    "WARNING":  "🟠",
    "MONITOR":  "🔵",
    "NORMAL":   "🟢",
}

DEMO_CLIP_ICONS = {
    "Braking": "gauge",
    "Start-Up": "zap",
    "Idle": "activity",
}

# ===========================================================================
# Visual layer — icons, CSS, and small HTML render helpers.
# Nothing in this section touches inference, scoring, or session state.
# ===========================================================================

_ICON_PATHS = {
    "logo": '<path d="M2 12h4l2-7 4 14 2-7h4"/>',
    "wifi-off": '<path d="M12 20h.01"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/><path d="M5 12.859a10 10 0 0 1 5.17-2.69"/><path d="M19 12.859a10 10 0 0 0-2.007-1.523"/><path d="M2 8.82a15 15 0 0 1 4.177-2.643"/><path d="M22 8.82a15 15 0 0 0-11.288-3.764"/><path d="m2 2 20 20"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M9 2v2"/><path d="M15 20v2"/><path d="M9 20v2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M2 15h2"/><path d="M2 9h2"/>',
    "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.35 8.95a1 1 0 0 1-.65.01C8.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18.469"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18.469"/>',
    "mic": '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    "file-audio": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><polyline points="15 2 15 7 20 7"/><path d="M10 13a1 1 0 0 0-1 1v3a1 1 0 0 0 2 0v-3a1 1 0 0 0-1-1Z"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "upload-cloud": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "volume-2": '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><path d="M16 9a5 5 0 0 1 0 6"/><path d="M19.364 18.364a9 9 0 0 0 0-12.728"/>',
    "volume-x": '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><line x1="22" x2="16" y1="9" y2="15"/><line x1="16" x2="22" y1="9" y2="15"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>',
    "car": '<path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 2) -> str:
    path = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" class="revio-icon">{path}</svg>'
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @keyframes revioFadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes revioPulse {
            0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
            70%  { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes revioRingIn {
            from { opacity: 0; transform: scale(.85); }
            to   { opacity: 1; transform: scale(1); }
        }
        @keyframes revioGlow {
            0%, 100% { box-shadow: 0 0 0 1px rgba(59,130,246,.35), 0 0 18px rgba(59,130,246,.18); }
            50%      { box-shadow: 0 0 0 1px rgba(59,130,246,.55), 0 0 26px rgba(59,130,246,.32); }
        }

        html, body, [class*="css"] {
            font-family: -apple-system, "Segoe UI Variable", "Segoe UI", "Inter", system-ui, Roboto, sans-serif !important;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(59,130,246,.10), transparent 60%),
                radial-gradient(900px 500px at 110% 10%, rgba(16,185,129,.06), transparent 55%),
                #0B1120;
        }

        header[data-testid="stHeader"] { background: transparent; }
        div.block-container { padding-top: 1.1rem; max-width: 1360px; }

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

        h1, h2, h3, h4 { color: #F8FAFC !important; letter-spacing: -0.01em; }
        p, span, label, .stMarkdown { color: #F8FAFC; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }

        hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.1rem 0 !important; }

        /* ---- Bordered containers used as "cards" ---------------------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
            animation: revioFadeUp .35s ease both;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
        }
        div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
        div.block-container > div [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, #1A2234 0%, #171F30 100%) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
            transition: box-shadow .25s ease, transform .25s ease, border-color .25s ease;
        }
        div.block-container [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(59,130,246,0.35) !important;
            box-shadow: 0 10px 26px rgba(0,0,0,0.32);
        }

        /* ---- Buttons ---------------------------------------------------- */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: .01em;
            border: 1px solid rgba(255,255,255,0.10);
            background: #1A2234;
            color: #F8FAFC;
            transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(59,130,246,0.45);
        }
        .stButton > button:active { transform: translateY(0); }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            border: none;
            box-shadow: 0 8px 20px rgba(59,130,246,0.35);
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 26px rgba(59,130,246,0.5);
            transform: translateY(-2px);
        }
        .stButton > button:disabled {
            opacity: .45;
            box-shadow: none;
            transform: none;
        }

        /* ---- Inputs: selects, radios, toggle ---------------------------- */
        div[data-baseweb="select"] > div {
            background: #1A2234 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            transition: border-color .2s ease, box-shadow .2s ease;
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(59,130,246,0.55) !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
        }
        ul[data-testid="stSelectboxVirtualDropdown"] { animation: revioFadeUp .18s ease both; }

        div[role="radiogroup"] label {
            background: #1A2234;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 8px 14px !important;
            margin-right: 8px !important;
            transition: border-color .2s ease, box-shadow .2s ease, transform .15s ease;
        }
        div[role="radiogroup"] label:hover { border-color: rgba(59,130,246,0.4); transform: translateY(-1px); }
        div[role="radiogroup"] label[data-checked="true"] {
            border-color: #3B82F6;
            box-shadow: 0 0 0 1px rgba(59,130,246,.35), 0 0 16px rgba(59,130,246,.18);
        }

        /* Demo-clip card grid (scoped via container key) */
        div.st-key-demo_clip_cards div[role="radiogroup"] { flex-direction: column; gap: 8px; }
        div.st-key-demo_clip_cards div[role="radiogroup"] label {
            width: 100%; padding: 12px 16px !important; border-radius: 14px;
        }
        div.st-key-demo_clip_cards div[role="radiogroup"] label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(59,130,246,.16), rgba(59,130,246,.05));
            animation: revioGlow 2.4s ease-in-out infinite;
        }

        label[data-baseweb="checkbox"] { color: #F8FAFC; }

        /* ---- File uploader ------------------------------------------------ */
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, #141b2c 0%, #111827 100%) !important;
            border: 1.5px dashed rgba(59,130,246,0.35) !important;
            border-radius: 16px !important;
            transition: border-color .2s ease, background .2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(59,130,246,0.7) !important;
            background: linear-gradient(180deg, #16203a 0%, #111827 100%) !important;
        }

        /* ---- Expander -------------------------------------------------- */
        details[data-testid="stExpander"], div[data-testid="stExpander"] {
            background: #1A2234;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px !important;
            overflow: hidden;
        }

        /* ---- Dataframe --------------------------------------------------- */
        [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }

        /* ---- Alerts / info boxes ------------------------------------------ */
        div[data-testid="stAlertContainer"] {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }

        /* =================== Custom component classes =================== */
        .revio-navbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 14px 22px; border-radius: 18px; margin-bottom: 18px;
            background: linear-gradient(135deg, #111827 0%, #0d1420 100%);
            border: 1px solid rgba(255,255,255,0.08);
            animation: revioFadeUp .3s ease both;
        }
        .revio-navbar-left { display: flex; align-items: center; gap: 12px; }
        .revio-logo-mark {
            width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #3B82F6, #1d4ed8); color: white;
            box-shadow: 0 6px 16px rgba(59,130,246,.4);
        }
        .revio-brand-title { font-size: 1.28rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; }
        .revio-brand-sub { font-size: .78rem; color: #94A3B8; font-weight: 500; }
        .revio-navbar-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

        .revio-chip {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 12px; border-radius: 999px; font-size: .76rem; font-weight: 600;
            border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.03); color: #F8FAFC;
            white-space: nowrap;
        }
        .revio-chip svg { flex-shrink: 0; }
        .revio-chip-offline { color: #10B981; border-color: rgba(16,185,129,.35); background: rgba(16,185,129,.08); }
        .revio-chip-blue { color: #93C5FD; border-color: rgba(59,130,246,.35); background: rgba(59,130,246,.08); }
        .revio-chip-neutral { color: #94A3B8; }

        .revio-hero {
            display: flex; align-items: center; justify-content: space-between; gap: 24px;
            flex-wrap: wrap;
            padding: 30px 32px; border-radius: 20px; margin-bottom: 20px;
            background: linear-gradient(120deg, rgba(59,130,246,0.10), rgba(17,24,39,0.4) 55%), #111827;
            border: 1px solid rgba(255,255,255,0.08);
            animation: revioFadeUp .4s ease both;
        }
        .revio-hero-title { font-size: 1.9rem; font-weight: 800; color: #F8FAFC; margin: 0 0 6px 0; letter-spacing: -.02em; }
        .revio-hero-sub { font-size: .96rem; color: #94A3B8; margin: 0; max-width: 520px; }
        .revio-hero-kpis { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }

        .revio-pulse-dot {
            width: 9px; height: 9px; border-radius: 50%; background: #10B981;
            animation: revioPulse 2s infinite; display: inline-block;
        }

        .revio-kpi {
            min-width: 118px; padding: 10px 14px; border-radius: 14px;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        }
        .revio-kpi-label { font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; color: #94A3B8; font-weight: 600; }
        .revio-kpi-value { font-size: 1.02rem; font-weight: 700; color: #F8FAFC; margin-top: 2px; }

        .revio-section-title {
            display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: .92rem;
            color: #F8FAFC; margin-bottom: 2px;
        }
        .revio-section-title svg { color: #3B82F6; }
        .revio-section-desc { font-size: .76rem; color: #94A3B8; margin-bottom: 10px; }

        .revio-status-card {
            display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 12px;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 8px;
        }
        .revio-status-dot {
            width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
            background: #10B981; animation: revioPulse 2.4s infinite;
        }
        .revio-status-dot.off { background: #64748B; animation: none; }
        .revio-status-text { font-size: .82rem; font-weight: 600; color: #F8FAFC; }
        .revio-status-desc { font-size: .72rem; color: #94A3B8; }

        .revio-empty {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            text-align: center; padding: 52px 24px; border-radius: 18px;
            border: 1.5px dashed rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.015);
            color: #94A3B8;
        }
        .revio-empty svg { color: #3B82F6; opacity: .55; margin-bottom: 14px; }
        .revio-empty-title { font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px; }
        .revio-empty-sub { font-size: .82rem; max-width: 320px; }

        .revio-risk-badge {
            display: flex; align-items: center; justify-content: center; gap: 10px;
            padding: 16px 22px; border-radius: 16px; margin-bottom: 14px;
            font-size: 1.3rem; font-weight: 800; color: white; letter-spacing: .02em;
            animation: revioFadeUp .3s ease both;
        }

        .revio-result-grid { display: flex; gap: 22px; flex-wrap: wrap; align-items: center; margin-bottom: 6px; }
        .revio-ring-wrap { display: flex; flex-direction: column; align-items: center; animation: revioRingIn .35s ease both; }
        .revio-ring-value { font-size: 1.5rem; font-weight: 800; fill: #F8FAFC; }
        .revio-ring-label { font-size: .72rem; color: #94A3B8; margin-top: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }

        .revio-fact-list { flex: 1; min-width: 220px; display: flex; flex-direction: column; gap: 10px; }
        .revio-fact-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 12px; border-radius: 10px; background: rgba(255,255,255,0.03); }
        .revio-fact-label { font-size: .78rem; color: #94A3B8; font-weight: 600; }
        .revio-fact-value { font-size: .86rem; color: #F8FAFC; font-weight: 700; }

        .revio-prob-bar-row { margin-bottom: 10px; }
        .revio-prob-bar-head { display: flex; justify-content: space-between; font-size: .78rem; margin-bottom: 4px; }
        .revio-prob-bar-track { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
        .revio-prob-bar-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }

        .revio-rec-card {
            display: flex; gap: 14px; padding: 16px 18px; border-radius: 16px; margin: 14px 0;
            border: 1px solid rgba(255,255,255,0.08);
            animation: revioFadeUp .35s ease both;
        }
        .revio-rec-icon { flex-shrink: 0; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .revio-rec-title { font-weight: 700; font-size: .88rem; margin-bottom: 3px; }
        .revio-rec-desc { font-size: .82rem; color: #CBD5E1; line-height: 1.45; }

        .revio-metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 4px; }
        .revio-metric-tile {
            flex: 1; min-width: 130px; padding: 12px 14px; border-radius: 12px;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        }
        .revio-metric-tile-label { font-size: .68rem; color: #94A3B8; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
        .revio-metric-tile-value { font-size: 1.0rem; font-weight: 700; color: #F8FAFC; margin-top: 3px; }

        .revio-footer {
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
            padding: 16px 22px; margin-top: 26px; border-radius: 16px;
            background: #111827; border: 1px solid rgba(255,255,255,0.08);
            font-size: .76rem; color: #94A3B8;
        }
        .revio-footer b { color: #F8FAFC; }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 8px; }

        *:focus-visible { outline: 2px solid #3B82F6 !important; outline-offset: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def circular_progress(value_pct: float, color: str, size: int = 128, stroke: int = 10) -> str:
    """Animated SVG ring for a 0-100 confidence value. Purely presentational."""
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - max(0.0, min(1.0, value_pct / 100.0)))
    center = size / 2
    return f"""
    <div class="revio-ring-wrap">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="rgba(255,255,255,0.08)" stroke-width="{stroke}"/>
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="{color}" stroke-width="{stroke}" stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 {center} {center})"
                style="transition: stroke-dashoffset 1s cubic-bezier(.4,0,.2,1);"/>
        <text x="50%" y="50%" text-anchor="middle" dy="0.35em" class="revio-ring-value">{value_pct:.0f}%</text>
      </svg>
      <div class="revio-ring-label">Confidence</div>
    </div>
    """.strip()


def probability_bar_row(label: str, prob: float, color: str) -> str:
    pct = prob * 100
    return f"""
    <div class="revio-prob-bar-row">
      <div class="revio-prob-bar-head">
        <span style="color:#F8FAFC;font-weight:600;">{label}</span>
        <span style="color:{color};font-weight:700;">{pct:.1f}%</span>
      </div>
      <div class="revio-prob-bar-track">
        <div class="revio-prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
      </div>
    </div>
    """.strip()


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classifier(state: str) -> TFLiteClassifier:
    cfg = STATE_CONFIG[state]
    return TFLiteClassifier(str(cfg["model"]), cfg["labels"])

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "is_muted" not in st.session_state:
    st.session_state.is_muted = False
if "auto_state" not in st.session_state:
    st.session_state.auto_state = list(STATE_CONFIG.keys())[0]
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # UI-only cache of the most recent inference, for KPI display

inject_css()

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------
last = st.session_state.last_run
latency_chip = f"{last['latency_ms']:.1f} ms" if last else "—"

st.markdown(
    f"""
    <div class="revio-navbar">
      <div class="revio-navbar-left">
        <div class="revio-logo-mark">{icon('logo', 22, 'white', 2.4)}</div>
        <div>
          <div class="revio-brand-title">Revio</div>
          <div class="revio-brand-sub">Edge AI Vehicle Health Monitor</div>
        </div>
      </div>
      <div class="revio-navbar-right">
        <span class="revio-chip revio-chip-offline">{icon('wifi-off', 14, '#10B981')} Offline</span>
        <span class="revio-chip revio-chip-blue">{icon('cpu', 14, '#93C5FD')} TFLite Engine</span>
        <span class="revio-chip revio-chip-neutral">{icon('layers', 14, '#94A3B8')} {st.session_state.auto_state} Model</span>
        <span class="revio-chip revio-chip-neutral">{icon('settings', 14, '#94A3B8')}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="revio-hero">
      <div>
        <p class="revio-hero-title">Edge AI Vehicle Health Monitor</p>
        <p class="revio-hero-sub">Real-time on-device acoustic diagnostics powered by Edge AI — fully offline, no cloud dependency.</p>
      </div>
      <div class="revio-hero-kpis">
        <div class="revio-kpi">
          <div class="revio-kpi-label"><span class="revio-pulse-dot"></span>&nbsp; Status</div>
          <div class="revio-kpi-value">Online (Edge)</div>
        </div>
        <div class="revio-kpi">
          <div class="revio-kpi-label">Model Loaded</div>
          <div class="revio-kpi-value">{st.session_state.auto_state}</div>
        </div>
        <div class="revio-kpi">
          <div class="revio-kpi-label">Last Latency</div>
          <div class="revio-kpi-value">{latency_chip}</div>
        </div>
        <div class="revio-kpi">
          <div class="revio-kpi-label">Inference</div>
          <div class="revio-kpi-value">{'Complete' if last else 'Idle'}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="revio-section-title">{icon("sliders", 16)} Controls</div>'
        f'<div class="revio-section-desc">Configure vehicle state and alerting</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("car", 16)} Vehicle State</div>',
            unsafe_allow_html=True,
        )
        # Vehicle state selector (simulates OBD signal)
        # auto_state is updated automatically when a demo clip is selected
        state_options = list(STATE_CONFIG.keys())
        state = st.selectbox(
            "Vehicle State (OBD signal)",
            state_options,
            index=state_options.index(st.session_state.auto_state),
            key="state_selector",
            help="In a real deployment this is determined by RPM, brake switch, and speed — not ML.",
            label_visibility="collapsed",
        )
        st.caption(f"_{STATE_CONFIG[state]['description']}_")

    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("volume-2" if not st.session_state.is_muted else "volume-x", 16)} Voice Alerts</div>',
            unsafe_allow_html=True,
        )
        # Mute toggle
        st.session_state.is_muted = st.toggle(
            "Mute spoken alerts",
            value=st.session_state.is_muted,
        )
        st.caption("CRITICAL and WARNING tiers auto-speak when unmuted.")

    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("shield", 16)} System Health</div>',
            unsafe_allow_html=True,
        )
        tts_status = "Muted" if st.session_state.is_muted else "pyttsx3 (offline)"
        health_rows = [
            ("cpu",          "TFLite Inference",  "On-device", True),
            ("activity",     "MFCC Extraction",   "librosa",   True),
            ("brain",        "Rule Engine",       "Risk scoring", True),
            ("wifi-off",     "Offline Mode",      "No network calls", True),
            ("volume-2" if not st.session_state.is_muted else "volume-x", "TTS", tts_status, not st.session_state.is_muted),
        ]
        rows_html = "".join(
            f'<div class="revio-status-card">'
            f'<span class="revio-status-dot{"" if on else " off"}"></span>'
            f'{icon(ic, 15, "#94A3B8")}'
            f'<div><div class="revio-status-text">{name}</div>'
            f'<div class="revio-status-desc">{desc}</div></div>'
            f'</div>'
            for ic, name, desc, on in health_rows
        )
        st.markdown(rows_html, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("trash", 16)} Session</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_run = None
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — two columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("file-audio", 18)} Audio Input</div>'
            f'<div class="revio-section-desc">Provide a recording to analyse</div>',
            unsafe_allow_html=True,
        )

        input_mode = st.radio("Source", ["Upload audio clip", "Use demo clip"], horizontal=True,
                               label_visibility="collapsed")
        audio_path = None
        audio_bytes = None

        if input_mode == "Upload audio clip":
            uploaded = st.file_uploader("Upload a .wav file", type=["wav"])
            if uploaded:
                audio_bytes = uploaded.read()
                tmp_path = ROOT / "data" / "samples" / uploaded.name
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(audio_bytes)
                audio_path = tmp_path
                st.markdown(
                    f'<div class="revio-status-card" style="margin-top:8px;">'
                    f'{icon("check-circle", 16, "#10B981")}'
                    f'<div><div class="revio-status-text">{uploaded.name}</div>'
                    f'<div class="revio-status-desc">{len(audio_bytes)/1024:.1f} KB · WAV</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.audio(audio_bytes, format="audio/wav")

        else:
            clip_names = list(DEMO_CLIPS.keys())
            display_names = [f"{DEMO_CLIPS[c][0]}  ({DEMO_CLIPS[c][1]})" for c in clip_names]

            with st.container(key="demo_clip_cards"):
                selected_idx = st.radio(
                    "Select demo clip", range(len(clip_names)),
                    format_func=lambda i: display_names[i],
                    label_visibility="collapsed",
                )
            selected_clip = clip_names[selected_idx]
            audio_path = DEMO_CLIPS_DIR / selected_clip
            recommended_state = DEMO_CLIPS[selected_clip][1]

            # Auto-sync the vehicle state selector to match the chosen clip
            if st.session_state.auto_state != recommended_state:
                st.session_state.auto_state = recommended_state
                st.rerun()

            st.caption(f"State auto-set to **{recommended_state}** to match this clip.")
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")

        run = st.button("Run Detection", type="primary", use_container_width=True,
                        disabled=(audio_path is None))

# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
with right_col:
    with st.container(border=True):
        st.markdown(
            f'<div class="revio-section-title">{icon("gauge", 18)} Detection Result</div>'
            f'<div class="revio-section-desc">Model output, risk, and recommended action</div>',
            unsafe_allow_html=True,
        )

        if run and audio_path is not None:
            with st.spinner("Extracting features and running inference..."):

                # 1 — MFCC extraction
                t0 = time.perf_counter()
                audio = load_audio(str(audio_path))
                mfcc  = extract_mfcc(audio)
                mfcc_time_ms = (time.perf_counter() - t0) * 1000

                # 2 — TFLite inference
                clf    = load_classifier(state)
                result = clf.predict(mfcc)

                fault_label = result["fault_label"]
                confidence  = result["confidence"]
                latency_ms  = result["latency_ms"]

                # 3 — Risk scoring
                risk = risk_score(fault_label, confidence)

                # 4 — Narrative
                narrative = get_narrative(fault_label, risk.risk_tier)

                # 5 — TTS (CRITICAL / WARNING only, respects mute)
                tts_fired = tts_alert(narrative, risk.risk_tier,
                                      is_muted=st.session_state.is_muted)

            # ── Risk badge ──────────────────────────────────────────────────
            tier   = risk.risk_tier
            colour = TIER_COLOURS[tier]
            emoji  = TIER_EMOJI[tier]

            st.markdown(
                f'<div class="revio-risk-badge" style="background:linear-gradient(135deg,{colour}dd,{colour}aa);">'
                f'{emoji} {tier}</div>',
                unsafe_allow_html=True,
            )

            # ── Confidence ring + key facts ──────────────────────────────────
            ring_html = circular_progress(confidence * 100, colour)
            facts_html = f"""
            <div class="revio-fact-list">
              <div class="revio-fact-row"><span class="revio-fact-label">Fault Category</span>
                <span class="revio-fact-value">{fault_label.replace('_', ' ').title()}</span></div>
              <div class="revio-fact-row"><span class="revio-fact-label">Risk Level</span>
                <span class="revio-fact-value" style="color:{colour};">{tier}</span></div>
              <div class="revio-fact-row"><span class="revio-fact-label">Failure Probability</span>
                <span class="revio-fact-value">{risk.failure_probability:.1f}%</span></div>
            </div>
            """.strip()
            st.markdown(f'<div class="revio-result-grid">{ring_html}{facts_html}</div>', unsafe_allow_html=True)

            # Failure probability bar (visual only, same value used above)
            st.markdown(probability_bar_row("Failure probability", risk.failure_probability / 100, colour),
                        unsafe_allow_html=True)

            # ── Recommendation / narrative card ──────────────────────────────
            if tts_fired:
                alert_note = "🔊 Spoken alert played"
            elif st.session_state.is_muted:
                alert_note = "🔇 Spoken alert muted"
            else:
                alert_note = "Silent — NORMAL and MONITOR tiers do not trigger a spoken alert"

            rec_icon = "alert-triangle" if tier in (TIER_CRITICAL, TIER_WARNING) else "check-circle"
            st.markdown(
                f"""
                <div class="revio-rec-card" style="background:{colour}14; border-color:{colour}40;">
                  <div class="revio-rec-icon" style="background:{colour}22;">{icon(rec_icon, 20, colour)}</div>
                  <div>
                    <div class="revio-rec-title" style="color:{colour};">Driver Alert</div>
                    <div class="revio-rec-desc">{narrative}</div>
                    <div class="revio-rec-desc" style="margin-top:6px;opacity:.8;">{alert_note}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Pipeline + audio metrics ─────────────────────────────────────
            feature_count = int(mfcc.shape[0] * mfcc.shape[1])
            st.markdown(
                f"""
                <div class="revio-metric-row">
                  <div class="revio-metric-tile"><div class="revio-metric-tile-label">Inference Time</div>
                    <div class="revio-metric-tile-value">{latency_ms:.2f} ms</div></div>
                  <div class="revio-metric-tile"><div class="revio-metric-tile-label">Model Used</div>
                    <div class="revio-metric-tile-value">{state}</div></div>
                  <div class="revio-metric-tile"><div class="revio-metric-tile-label">Audio Length</div>
                    <div class="revio-metric-tile-value">{DURATION:.1f}s</div></div>
                  <div class="revio-metric-tile"><div class="revio-metric-tile-label">Feature Count</div>
                    <div class="revio-metric-tile-value">{feature_count}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Pipeline timings"):
                st.markdown(f"- MFCC extraction : **{mfcc_time_ms:.1f} ms**")
                st.markdown(f"- TFLite inference: **{latency_ms:.2f} ms**")
                st.markdown(f"- Total           : **{mfcc_time_ms + latency_ms:.1f} ms**")
                st.markdown("_All processing runs on-device — zero network calls_")

            with st.expander("All class probabilities"):
                for label, prob in sorted(result["all_probabilities"].items(),
                                           key=lambda x: -x[1]):
                    bar_colour = colour if label == fault_label else "#3B82F6"
                    st.markdown(probability_bar_row(label.replace("_", " ").title(), prob, bar_colour),
                                unsafe_allow_html=True)

            # ── Cache for hero/status KPIs (UI-only, does not affect history) ──
            st.session_state.last_run = {
                "latency_ms": latency_ms,
                "state": state,
                "tier": tier,
            }

            # ── Add to history ───────────────────────────────────────────────
            st.session_state.history.append({
                "time":       time.strftime("%H:%M:%S"),
                "state":      state,
                "fault":      fault_label.replace("_", " "),
                "tier":       tier,
                "confidence": f"{confidence:.1%}",
                "fail_prob":  f"{risk.failure_probability:.1f}%",
            })

        else:
            st.markdown(
                f"""
                <div class="revio-empty">
                    {icon('mic', 44, '#3B82F6', 1.6)}
                    <div class="revio-empty-title">Ready for Analysis</div>
                    <div class="revio-empty-sub">Select or upload an audio clip, then press
                    <b>Run Detection</b> to run the on-device pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Session history log
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        f'<div class="revio-section-title">{icon("clock", 18)} Detection History (this session)</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)

        # Colour the Tier column
        def colour_tier(val):
            c = TIER_COLOURS.get(val, "#888")
            return f"background-color:{c};color:white;font-weight:bold;border-radius:4px;"

        styled = df.style.map(colour_tier, subset=["tier"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"""
            <div class="revio-empty" style="padding:28px;">
                {icon('database', 30, '#3B82F6', 1.6)}
                <div class="revio-empty-title" style="font-size:.92rem;">No detections yet</div>
                <div class="revio-empty-sub">Results from this session will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="revio-footer">
      <div><b>Revio</b> Edge AI Vehicle Health Monitor · Model v1.0</div>
      <div>Inference Engine: <b>TensorFlow Lite</b></div>
      <div><b>100% Offline</b> · On-device processing</div>
      <div>Build {time.strftime('%Y.%m.%d')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
