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

from src.preprocess import load_audio, extract_mfcc
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

TIER_COLOURS = {
    "CRITICAL": "#E74C3C",
    "WARNING":  "#E67E22",
    "MONITOR":  "#F1C40F",
    "NORMAL":   "#2ECC71",
}

TIER_EMOJI = {
    "CRITICAL": "🔴",
    "WARNING":  "🟠",
    "MONITOR":  "🟡",
    "NORMAL":   "🟢",
}

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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("🔊 Revio")
    st.markdown("**Edge AI Vehicle Health Monitor** — *fully offline, no cloud dependency*")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🌐 Running fully offline — no cloud dependency")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    # Vehicle state selector (simulates OBD signal)
    # auto_state is updated automatically when a demo clip is selected
    state_options = list(STATE_CONFIG.keys())
    state = st.selectbox(
        "Vehicle State (OBD signal)",
        state_options,
        index=state_options.index(st.session_state.auto_state),
        key="state_selector",
        help="In a real deployment this is determined by RPM, brake switch, and speed — not ML.",
    )
    st.caption(f"_{STATE_CONFIG[state]['description']}_")

    st.divider()

    # Mute toggle
    st.session_state.is_muted = st.toggle(
        "Mute spoken alerts",
        value=st.session_state.is_muted,
    )
    st.caption("CRITICAL and WARNING tiers auto-speak when unmuted.")

    st.divider()

    # Offline indicator
    st.markdown("**System status**")
    st.markdown("- ✅ TFLite inference (on-device)")
    st.markdown("- ✅ MFCC extraction (librosa)")
    st.markdown("- ✅ Risk scoring (rule-based)")
    st.markdown("- ✅ Narratives (template-based)")
    tts_status = "🔇 Muted" if st.session_state.is_muted else "✅ pyttsx3 (offline)"
    st.markdown(f"- {tts_status} TTS")

    st.divider()
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — two columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Audio Input")

    input_mode = st.radio("Source", ["Upload audio clip", "Use demo clip"], horizontal=True)
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
            st.audio(audio_bytes, format="audio/wav")

    else:
        clip_names = list(DEMO_CLIPS.keys())
        display_names = [f"{DEMO_CLIPS[c][0]}  ({DEMO_CLIPS[c][1]})" for c in clip_names]
        selected_idx = st.selectbox("Select demo clip", range(len(clip_names)),
                                     format_func=lambda i: display_names[i])
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
    st.subheader("Detection Result")

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

        # ── Risk badge ──────────────────────────────────────────────────────
        tier   = risk.risk_tier
        colour = TIER_COLOURS[tier]
        emoji  = TIER_EMOJI[tier]

        st.markdown(
            f"""
            <div style="background:{colour};padding:18px 24px;border-radius:12px;
                        text-align:center;margin-bottom:12px;">
              <span style="font-size:2rem;font-weight:900;color:white;">
                {emoji} {tier}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Fault label (full width — avoids truncation in narrow metric column)
        st.markdown(
            f"**Fault detected:** `{fault_label.replace('_', ' ').title()}`",
            unsafe_allow_html=False,
        )

        # ── Metrics row ──────────────────────────────────────────────────────
        m2, m3 = st.columns(2)
        m2.metric("Confidence",          f"{confidence:.1%}")
        m3.metric("Failure probability", f"{risk.failure_probability:.1f}%")

        # ── Narrative ────────────────────────────────────────────────────────
        st.info(f"**Driver alert:** {narrative}")

        if tts_fired:
            st.caption("🔊 Spoken alert played")
        elif st.session_state.is_muted:
            st.caption("🔇 Spoken alert muted")
        else:
            st.caption("Silent — NORMAL and MONITOR tiers do not trigger spoken alert")

        # ── Pipeline timings ─────────────────────────────────────────────────
        with st.expander("Pipeline timings"):
            st.markdown(f"- MFCC extraction : **{mfcc_time_ms:.1f} ms**")
            st.markdown(f"- TFLite inference: **{latency_ms:.2f} ms**")
            st.markdown(f"- Total           : **{mfcc_time_ms + latency_ms:.1f} ms**")
            st.markdown("_All processing runs on-device — zero network calls_")

        # ── All class probabilities ───────────────────────────────────────────
        with st.expander("All class probabilities"):
            for label, prob in sorted(result["all_probabilities"].items(),
                                       key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                st.markdown(
                    f"`{label:<30}` **{prob:.3f}** {bar}"
                )

        # ── Add to history ───────────────────────────────────────────────────
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
            """
            <div style="border:2px dashed #aaa;border-radius:12px;padding:40px;
                        text-align:center;color:#888;">
                Select or upload an audio clip and press<br>
                <strong>Run Detection</strong> to start.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Session history log
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Detection History (this session)")

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
    st.caption("No detections yet this session.")
