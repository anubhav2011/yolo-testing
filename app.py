"""
Streamlit web UI: upload video, run proctoring, download JSON results.
"""

import json
import os
import sys
import tempfile

import streamlit as st

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from main import ProctoringAnalyzer

st.set_page_config(page_title="Proctoring", page_icon="📹", layout="centered")

if "proctoring_result" not in st.session_state:
    st.session_state.proctoring_result = None
if "proctoring_error" not in st.session_state:
    st.session_state.proctoring_error = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

st.title("Proctoring (YOLO + Pose)")
st.caption("Upload a video, then start analysis to get a gesture report as JSON.")

uploaded = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Supported: MP4, AVI, MOV, MKV, WebM",
)

if uploaded is None:
    st.session_state.proctoring_result = None
    st.session_state.proctoring_error = None
    st.session_state.last_upload_id = None

if uploaded is not None:
    upload_id = (uploaded.name, uploaded.size)
    if st.session_state.last_upload_id != upload_id:
        st.session_state.proctoring_result = None
        st.session_state.proctoring_error = None
        st.session_state.last_upload_id = upload_id

start = st.button("Start proctoring", type="primary", disabled=uploaded is None)

if start and uploaded is not None:
    st.session_state.proctoring_result = None
    st.session_state.proctoring_error = None

    CONFIG.VERBOSE = False
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    try:
        progress_bar = st.progress(0)
        status = st.empty()

        analyzer = ProctoringAnalyzer()

        def on_progress(current: int, total: int) -> None:
            if total and total > 0:
                p = min(float(current) / float(total), 1.0)
                progress_bar.progress(p)
                status.text(f"Processing frames: {current} / {total}")
            else:
                status.text("Processing…")

        with st.spinner("Running proctoring pipeline…"):
            result = analyzer.analyze_video(video_path, progress_callback=on_progress)

        progress_bar.progress(1.0)
        status.empty()

        if isinstance(result, dict) and result.get("error"):
            st.session_state.proctoring_error = result["error"]
            st.session_state.proctoring_result = None
        else:
            st.session_state.proctoring_error = None
            st.session_state.proctoring_result = result
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass

if st.session_state.proctoring_error:
    st.error(st.session_state.proctoring_error)

if st.session_state.proctoring_result is not None:
    st.success("Proctoring is completed.")
    json_str = json.dumps(
        st.session_state.proctoring_result,
        indent=CONFIG.OUTPUT_JSON_INDENT,
    )
    st.download_button(
        label="Download results (JSON)",
        data=json_str,
        file_name="proctoring_results.json",
        mime="application/json",
        type="primary",
    )
    with st.expander("Preview JSON"):
        st.json(st.session_state.proctoring_result)

elif uploaded is None:
    st.info("Choose a video file to enable **Start proctoring**.")
