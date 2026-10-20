#!/usr/bin/env python3
"""Browser control panel for the direct-to-reverse parking sequence."""
import json
import html
import os
import subprocess
import time
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "gui_runs"
WORKER = ROOT / "07_streamlit_worker.py"
WORKER_PYTHON = os.environ.get("JETRACER_WORKER_PYTHON", "/usr/bin/python3")
CAPTURE_WIDTH = "1920"
CAPTURE_HEIGHT = "1080"
CAPTURE_FPS = "30"


def columns(count):
    function = getattr(st, "columns", None) or st.beta_columns
    return function(count)


def rerun():
    function = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if function:
        function()


def read_status(run_dir):
    if run_dir is None:
        return {"state": "idle", "stage": "Ready", "message": "Car is stopped"}
    path = Path(run_dir) / "status.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"state": "starting", "stage": "Starting", "message": "Waiting for worker"}


def start_run():
    RUNS.mkdir(exist_ok=True)
    run_dir = RUNS / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    environment = dict(os.environ)
    environment.update(JETRACER_CAPTURE_WIDTH=CAPTURE_WIDTH,
                       JETRACER_CAPTURE_HEIGHT=CAPTURE_HEIGHT,
                       JETRACER_CAPTURE_FPS=CAPTURE_FPS,
                       PYTHONUNBUFFERED="1")
    log = (run_dir / "worker.log").open("w")
    process = subprocess.Popen(
        [WORKER_PYTHON, str(WORKER), str(run_dir)], cwd=str(ROOT), env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    st.session_state.process = process
    st.session_state.run_dir = str(run_dir)


def stop_run():
    process = st.session_state.get("process")
    if process is not None and process.poll() is None:
        process.terminate()


st.set_page_config(page_title="JetRacer Parking", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(145deg, #07111f 0%, #0c1a2d 48%, #101b2a 100%); color: #e2e8f0;}
.block-container {max-width: 1240px; padding-top: 1.25rem; padding-bottom: 2rem;}
.hero {padding: 1.6rem 1.8rem; border: 1px solid rgba(125,211,252,.18);
       border-radius: 22px; margin-bottom: 1rem;
       background: radial-gradient(circle at 85% 10%, rgba(14,165,233,.2), transparent 34%), #0b1627;
       box-shadow: 0 18px 50px rgba(0,0,0,.25);}
.eyebrow {color:#38bdf8; font-size:.72rem; font-weight:800; letter-spacing:.18em;}
.hero h1 {color:#f8fafc; font-size:2.25rem; margin:.35rem 0 .3rem 0;}
.hero p {color:#94a3b8; margin:0; font-size:1rem;}
.status-card {padding: .9rem 1.15rem; border-radius: 14px; background: rgba(15,31,51,.9);
              border: 1px solid rgba(148,163,184,.18); margin:.5rem 0 1rem;}
.status-dot {height:9px; width:9px; border-radius:50%; background:#22c55e;
             display:inline-block; margin-right:8px; box-shadow:0 0 12px #22c55e;}
.metric-row {display:flex; gap:10px; margin:.2rem 0 1rem;}
.metric {flex:1; background:#0b1627; border:1px solid rgba(148,163,184,.15);
         border-radius:13px; padding:.8rem 1rem;}
.metric-label {font-size:.7rem; letter-spacing:.12em; color:#64748b; font-weight:800;}
.metric-value {font-size:1.05rem; color:#f1f5f9; font-weight:700; margin-top:3px;}
.telemetry {background:#0b1627; border:1px solid rgba(56,189,248,.16); border-radius:18px;
            padding:1.15rem 1.25rem; min-height:245px;}
.telemetry-title {font-size:.75rem; letter-spacing:.14em; font-weight:800; color:#7dd3fc; margin-bottom:1.1rem;}
.meter-label {display:flex; justify-content:space-between; color:#cbd5e1; font-size:.83rem; margin:.8rem 0 .35rem;}
.track {height:18px; border-radius:12px; background:#1e293b; position:relative; overflow:hidden;}
.track:after {content:''; position:absolute; left:50%; top:0; bottom:0; width:2px; background:#64748b;}
.steer-marker {position:absolute; top:2px; width:12px; height:14px; margin-left:-6px;
               border-radius:5px; background:#22d3ee; box-shadow:0 0 13px #22d3ee;}
.throttle-marker {position:absolute; top:2px; width:12px; height:14px; margin-left:-6px;
                  border-radius:5px; background:#4ade80; box-shadow:0 0 13px #4ade80;}
.scale {display:flex; justify-content:space-between; color:#64748b; font-size:.67rem; margin-top:3px;}
.small-note {color:#64748b; font-size:.86rem;}
div.stButton > button {width:100%; min-height:48px; border-radius:12px; font-weight:750;
                       border:1px solid rgba(56,189,248,.28);}
h2, h3 {color:#f1f5f9 !important;}
</style>
""", unsafe_allow_html=True)

if "process" not in st.session_state:
    st.session_state.process = None
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

process = st.session_state.process
active = process is not None and process.poll() is None
status = read_status(st.session_state.run_dir)

st.markdown("""
<div class="hero">
  <div class="eyebrow">JETRACER • AUTONOMOUS PARKING</div>
  <h1>Parking Control Room</h1>
  <p>Three start frames select the route: direct only, or direct → reverse. Every run records the camera and control telemetry.</p>
</div>
""", unsafe_allow_html=True)

left, right = columns(2)
with left:
    if st.button("▶  Start parking and record"):
        if active:
            st.warning("A parking run is already active.")
        else:
            start_run()
            rerun()
with right:
    if st.button("■ Stop car"):
        stop_run()
        st.warning("Stop requested. The worker is returning the motor command to zero.")

state = status.get("state", "starting")
stage = status.get("stage", "Starting")
message = status.get("message", "")
state_color = "#22c55e" if state == "running" else "#38bdf8"
if state == "error":
    state_color = "#fb7185"
st.markdown('<div class="status-card"><span class="status-dot" style="background:{}"></span>'
            '<b>{}</b><br><span style="color:#94a3b8">{}</span></div>'.format(
                state_color, html.escape(str(stage)), html.escape(str(message))),
            unsafe_allow_html=True)

st.markdown('<div class="metric-row"><div class="metric"><div class="metric-label">STATE</div>'
            '<div class="metric-value">{}</div></div><div class="metric"><div class="metric-label">STAGE</div>'
            '<div class="metric-value">{}</div></div><div class="metric"><div class="metric-label">RECORDED FRAMES</div>'
            '<div class="metric-value">{}</div></div></div>'.format(
                html.escape(state.title()), html.escape(str(stage)), status.get("frame_count", 0)),
            unsafe_allow_html=True)

camera, details = columns([2, 1])
run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
with camera:
    st.subheader("Live annotated camera")
    preview = run_dir / "preview.jpg" if run_dir else None
    if preview and preview.exists():
        st.image(str(preview), use_column_width=True)
    else:
        st.info("The live camera appears here after the run starts.")
with details:
    telemetry = status.get("telemetry") or {}
    steering = max(-1., min(1., float(telemetry.get("steering", 0.))))
    throttle = max(-1., min(1., float(telemetry.get("throttle", 0.))))
    steer_position = 50. + steering * 50.
    throttle_position = 50. + throttle * 50.
    st.markdown('<div class="telemetry"><div class="telemetry-title">LIVE CONTROL TELEMETRY</div>'
                '<div class="meter-label"><span>Steering</span><b>{:+.2f}</b></div>'
                '<div class="track"><div class="steer-marker" style="left:{:.1f}%"></div></div>'
                '<div class="scale"><span>LEFT</span><span>CENTER</span><span>RIGHT</span></div>'
                '<div class="meter-label"><span>Throttle</span><b>{:+.2f}</b></div>'
                '<div class="track"><div class="throttle-marker" style="left:{:.1f}%"></div></div>'
                '<div class="scale"><span>REVERSE</span><span>ZERO</span><span>FORWARD</span></div>'
                '<div style="color:#64748b;font-size:.78rem;margin-top:1.2rem">Step {} · {:.0f} ms inference · {:.0f} ms frame age</div>'
                '</div>'.format(steering, steer_position, throttle, throttle_position,
                                telemetry.get("step", 0), telemetry.get("inference_ms", 0),
                                telemetry.get("frame_age_ms", 0)), unsafe_allow_html=True)
    st.subheader("Run details")
    if run_dir:
        st.write("Run: `{}`".format(run_dir.name))
    result = status.get("result") or {}
    route = result.get("slot_decision") or status.get("slot_decision")
    if route:
        route_name = "Direct + reverse" if route.get("left_empty") else "Direct only"
        st.write("Selected route: **{}** (left-slot probability {:.1%})".format(
            route_name, float(route.get("left_probability", 0.))))
    if result.get("first"):
        st.write("Direct stage: **{}**, {} actions".format(
            result["first"]["reason"], result["first"]["steps"]))
    if result.get("second"):
        st.write("Reverse stage: **{}**, {} actions".format(
            result["second"]["reason"], result["second"]["steps"]))

video_name = status.get("video")
if video_name and Path(video_name).exists():
    st.subheader("Completed annotated run")
    st.video(str(video_name))
    st.caption("Saved at {}".format(video_name))
    telemetry_file = run_dir / "telemetry.csv" if run_dir else None
    if telemetry_file and telemetry_file.exists():
        st.caption("Frame telemetry saved at {}".format(telemetry_file))

if state == "error":
    st.error(status.get("message", "The worker stopped with an error."))
    with st.expander("Error details"):
        st.code(status.get("error", "No traceback was recorded."))

if run_dir and (run_dir / "worker.log").exists():
    with st.expander("Worker log"):
        lines = (run_dir / "worker.log").read_text(errors="replace").splitlines()
        st.code("\n".join(lines[-80:]))

st.markdown('<p class="small-note">Keep the browser open while testing. The Stop car button '
            'requests a neutral motor command from the hardware worker.</p>',
            unsafe_allow_html=True)

if active or state in ("starting", "running"):
    time.sleep(.5)
    rerun()
