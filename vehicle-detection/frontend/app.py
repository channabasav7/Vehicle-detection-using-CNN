"""
frontend/app.py

Streamlit web application for the Vehicle Type Detection and Classification
project. Provides: a dashboard, image detection, video detection, live
webcam detection, and an analytics page.

Run with:
    cd frontend
    streamlit run app.py
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.counting import VehicleCounter, keep_top_detections  # noqa: E402
from utils.model_loader import ModelLoadError, get_device_info, load_yolo_model  # noqa: E402

# --------------------------------------------------------------------------
# PAGE CONFIG + DARK THEME STYLING
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Detection & Classification",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232937 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h2 { color: #4ade80; margin: 0; font-size: 2.2rem; }
    .metric-card p { color: #9ca3af; margin: 4px 0 0 0; font-size: 0.9rem; }
    .section-card {
        background-color: #161b26;
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    h1, h2, h3 { color: #f3f4f6; }
    .stButton>button {
        background-color: #4ade80;
        color: #0e1117;
        font-weight: 600;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #22c55e; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
CLASS_NAMES_FALLBACK = ["auto", "bike", "bus", "car", "person", "tractor", "truck"]

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {timestamp, source_type, counts}
if "model" not in st.session_state:
    st.session_state.model = None


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def metric_card(col, value, label):
    col.markdown(
        f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_load_model(weights_path):
    return load_yolo_model(weights_path)


def get_model(weights_path):
    try:
        return cached_load_model(weights_path)
    except ModelLoadError as e:
        st.error(f"⚠️ {e}")
        return None


def draw_boxes(image_bgr, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names[cls_id] if isinstance(class_names, dict) else class_names[cls_id]
        label = f"{name}: {conf:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_bgr, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 200, 255), -1)
        cv2.putText(image_bgr, label, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image_bgr


def log_history(source_type, counts_dict):
    st.session_state.history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_type": source_type,
        **counts_dict,
    })


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.title("🚗 Vehicle Detection")
st.sidebar.caption("Deep Learning Vehicle Classification System")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Image Detection", "Video Detection", "Live Detection", "Analytics"],
)

st.sidebar.divider()
weights_path = st.sidebar.text_input("Model weights path", value=DEFAULT_WEIGHTS)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

device_info = get_device_info()
st.sidebar.info(f"Compute device: **{device_info['type']}** ({device_info['name']})")

# --------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Vehicle Type Detection and Classification")
    st.caption("AI-powered computer vision system built with YOLO, OpenCV, and Streamlit")

    model = get_model(weights_path)

    total_detections = sum(
        sum(v for k, v in row.items() if k in CLASS_NAMES_FALLBACK)
        for row in st.session_state.history
    )
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, len(st.session_state.history), "Sessions Logged")
    metric_card(c2, total_detections, "Total Vehicles Detected")
    metric_card(c3, len(CLASS_NAMES_FALLBACK), "Vehicle Classes")
    metric_card(c4, device_info["type"], "Compute Device")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📋 Model Information")
        if model is not None:
            class_names = model.names
            names_list = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
            st.write(f"**Weights:** `{weights_path}`")
            st.write(f"**Classes ({len(names_list)}):** {', '.join(names_list)}")
        else:
            st.write("Model not loaded. Set a valid weights path in the sidebar, "
                     "or train one with `training/train.py`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Vehicle Classes Supported")
        for name in CLASS_NAMES_FALLBACK:
            st.write(f"• {name.replace('_', ' ').title()}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏗️ System Architecture")
    st.code(
        "Input (Image / Video / Webcam)\n"
        "        |\n"
        "        v\n"
        "   Preprocessing\n"
        "        |\n"
        "        v\n"
        "  YOLO Deep Learning Model\n"
        "        |\n"
        "        v\n"
        "  Detection + Classification\n"
        "        |\n"
        "        v\n"
        " Bounding Box + Confidence\n"
        "        |\n"
        "        v\n"
        "   Vehicle Counting\n"
        "        |\n"
        "        v\n"
        " Analytics / Dashboard",
        language=None,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------------------------------
elif page == "Image Detection":
    st.title("🖼️ Image Detection")
    model = get_model(weights_path)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        col1.subheader("Original")
        col1.image(image, use_container_width=True)

        if st.button("🔍 Run Detection", key="run_image"):
            if model is None:
                st.stop()
            with st.spinner("Running detection..."):
                results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
                result = results[0]
                boxes = keep_top_detections(result.boxes, max_detections=1)
                class_names = model.names
                names_list = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)

                counter = VehicleCounter(names_list)
                for box in boxes:
                    counter.add_detection(int(box.cls[0]))

                annotated = draw_boxes(image_bgr.copy(), boxes, class_names)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col2.subheader("Detected")
            col2.image(annotated_rgb, use_container_width=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Vehicle Counts")
            summary = counter.summary()
            cols = st.columns(len(summary) + 1)
            for i, (name, count) in enumerate(summary.items()):
                metric_card(cols[i], count, name.replace("_", " ").title())
            metric_card(cols[-1], counter.total(), "Total")
            st.markdown("</div>", unsafe_allow_html=True)

            if counter.total() == 0:
                st.warning("No vehicles detected. Try lowering the confidence threshold in the sidebar.")
            else:
                log_history("image", counter.summary())

            out_dir = Path("../results")
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"detected_{uploaded.name}"
            cv2.imwrite(str(out_path), annotated)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated image", f, file_name=out_path.name)

# --------------------------------------------------------------------------
# VIDEO DETECTION
# --------------------------------------------------------------------------
elif page == "Video Detection":
    st.title("🎬 Video Detection")
    model = get_model(weights_path)

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None and model is not None:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp_in.write(uploaded.read())
        tmp_in.close()

        st.video(tmp_in.name)

        if st.button("🎥 Process Video"):
            cap = cv2.VideoCapture(tmp_in.name)
            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video. It may be corrupted or in an unsupported format.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            out_path = Path("../results/detected_video_streamlit.mp4")
            out_path.parent.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            class_names = model.names
            names_list = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
            counter = VehicleCounter(names_list)

            progress = st.progress(0, text="Processing frames...")
            frame_idx = 0
            preview_slot = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                result = results[0]
                boxes = result.boxes
                for box in boxes:
                    track_id = int(box.id[0]) if box.id is not None else None
                    counter.add_detection(int(box.cls[0]), track_id=track_id)

                annotated = draw_boxes(frame, boxes, class_names)
                writer.write(annotated)

                if frame_idx % 10 == 0 or frame_idx == n_frames:
                    preview_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        caption=f"Frame {frame_idx}/{n_frames}", use_container_width=True)
                progress.progress(min(frame_idx / max(n_frames, 1), 1.0))

            cap.release()
            writer.release()
            progress.empty()

            st.success("✅ Video processed.")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Unique Vehicle Counts")
            summary = counter.summary()
            cols = st.columns(len(summary) + 1)
            for i, (name, count) in enumerate(summary.items()):
                metric_card(cols[i], count, name.replace("_", " ").title())
            metric_card(cols[-1], counter.total(), "Total")
            st.markdown("</div>", unsafe_allow_html=True)

            log_history("video", counter.summary())
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download processed video", f, file_name=out_path.name)

# --------------------------------------------------------------------------
# LIVE DETECTION (WEBCAM)
# --------------------------------------------------------------------------
elif page == "Live Detection":
    st.title("📹 Live Webcam Detection")
    model = get_model(weights_path)

    st.info("Click **Start** to begin streaming from your webcam. Click **Stop** to end the session. "
            "This requires the app to run on a machine with a physically connected camera "
            "(browser-based webcam capture is not wired up in this reference build).")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_start, col_stop = st.columns(2)
    if col_start.button("▶️ Start"):
        st.session_state.webcam_running = True
    if col_stop.button("⏹️ Stop"):
        st.session_state.webcam_running = False

    frame_slot = st.empty()
    stats_slot = st.empty()

    if st.session_state.webcam_running and model is not None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam unavailable. Check it's connected, not used by another app, "
                     "and this process has camera permission.")
            st.session_state.webcam_running = False
        else:
            class_names = model.names
            names_list = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
            counter = VehicleCounter(names_list)
            t_start = time.time()
            frame_count = 0

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost webcam feed.")
                    break
                frame_count += 1

                results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                result = results[0]
                boxes = result.boxes
                for box in boxes:
                    track_id = int(box.id[0]) if box.id is not None else None
                    counter.add_detection(int(box.cls[0]), track_id=track_id)

                annotated = draw_boxes(frame, boxes, class_names)
                live_fps = frame_count / max(time.time() - t_start, 1e-6)
                cv2.putText(annotated, f"FPS: {live_fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                stats_slot.markdown(f"**Live count:** {counter.summary_text().replace(chr(10), ' | ')}")

            cap.release()
            if counter.total() > 0:
                log_history("webcam", counter.summary())

# --------------------------------------------------------------------------
# ANALYTICS
# --------------------------------------------------------------------------
elif page == "Analytics":
    st.title("📈 Analytics")

    if not st.session_state.history:
        st.info("No detection sessions logged yet. Run detection on an image, video, "
                 "or webcam session first — results will appear here.")
    else:
        df = pd.DataFrame(st.session_state.history).fillna(0)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Detection History")
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        class_cols = [c for c in df.columns if c in CLASS_NAMES_FALLBACK]
        totals = df[class_cols].sum().reset_index()
        totals.columns = ["vehicle_type", "count"]
        totals = totals[totals["count"] > 0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Vehicle Distribution")
            if not totals.empty:
                fig = px.pie(totals, names="vehicle_type", values="count", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(paper_bgcolor="#161b26", plot_bgcolor="#161b26",
                                   font_color="#f3f4f6")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No detections yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Detections Over Sessions")
            df["total"] = df[class_cols].sum(axis=1)
            fig2 = px.bar(df, x="timestamp", y="total", color="source_type",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(paper_bgcolor="#161b26", plot_bgcolor="#161b26",
                               font_color="#f3f4f6")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Export")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export detection history as CSV", csv, file_name="detection_history.csv")
        st.markdown("</div>", unsafe_allow_html=True)
