"""
================================================================================
  Smart Attendance System — Streamlit Web Application
================================================================================
  - Black & White Professional Theme
  - WebRTC Camera with Fixed Multi-Face Logic
  - Password Protected Dashboard
================================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import base64
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODULE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from face_detection import get_detector, detect_faces_in_frame
from embedding import get_facenet_model, generate_embedding, load_embedding_database
from recognition import recognize_face, parse_student_display_name, COSINE_SIMILARITY_THRESHOLD
from database import (
    init_database,
    insert_attendance,
    get_today_attendance,
    get_all_attendance,
    get_attendance_summary,
    clear_today_attendance,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System | Aror University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHING
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    """Load MTCNN detector and FaceNet model (reloaded per thread to avoid TF crashes)."""
    return get_detector(), get_facenet_model()

@st.cache_resource(show_spinner=False)
def load_database():
    """Load the prebuilt face embedding database (cached)."""
    return load_embedding_database()

# Models are loaded inside main() now

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_image(image_path):
    """Read an image and convert to base64 for CSS use."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMG_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

bg_base64 = get_base64_image(BG_IMG_PATH)
logo_base64 = get_base64_image(LOGO_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS / BRANDING (Black & White Theme)
# ─────────────────────────────────────────────────────────────────────────────
css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Full Page Background with Dark Overlay */
    .stApp {{
        background: linear-gradient(rgba(10, 10, 10, 0.9), rgba(15, 15, 15, 0.95)), 
                    url(data:image/jpeg;base64,{bg_base64});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 15, 15, 0.85) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Headers & Text */
    h1, h2, h3, p, span, div {{
        color: #e0e0e0;
    }}

    /* Header Container */
    .aror-header {{
        display: flex;
        align-items: center;
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }}
    .aror-header img {{
        margin-right: 20px;
        border-radius: 8px;
        background: rgba(255,255,255,0.1); /* Subtle background for contrast */
        padding: 5px;
    }}
    .aror-header h1 {{
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #ffffff;
    }}
    .aror-header p {{
        margin: 0;
        color: #aaaaaa;
        font-size: 1rem;
    }}

    /* Glass Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, background 0.3s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.05);
    }}
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ffffff;
    }}
    .metric-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #888888;
    }}

    /* Buttons */
    .stButton > button {{
        background: rgba(40, 40, 40, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        transform: translateY(-1px) !important;
    }}

    /* Results */
    .result-item {{
        background: rgba(30, 30, 30, 0.6);
        border-left: 4px solid #cccccc;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }}
    .result-item.unknown {{
        border-left-color: #555555;
    }}

    /* Hide elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def draw_results_on_frame(frame_bgr, face_results):
    for (x, y, w, h, name, confidence) in face_results:
        is_known = name != "Unknown"
        # B&W aesthetic bounding boxes
        color = (255, 255, 255) if is_known else (100, 100, 100) 

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

        _, display_name = parse_student_display_name(name) if is_known else ("", "Unknown")
        label = f"{display_name} ({confidence:.2f})" if is_known else "Unknown"

        # Safely position label: if face is too close to top edge, draw below the box
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y - 10
        if label_y - th < 0:
            label_y = y + h + th + 10

        cv2.rectangle(frame_bgr, (x, label_y - th - 5), (x + tw + 6, label_y + 5), color, -1)
        cv2.putText(frame_bgr, label, (x + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if is_known else (255, 255, 255), 2)
    return frame_bgr

def process_faces(rgb_frame, detector_model, facenet_model, db, is_upload=False):
    # Phase 7: Debugging Support - Logs
    def debug_log(msg):
        print(f"[DEBUG] {msg}")
        if is_upload:
            st.toast(f"🐛 {msg}")

    debug_log("Frame received for processing.")
    
    faces = detect_faces_in_frame(rgb_frame, detector_model)
    if faces:
        debug_log(f"Face detected: {len(faces)} found.")
    else:
        debug_log("No face detected in frame.")
        
    results = []
    for face in faces:
        x, y, w, h = face["box"]
        try:
            # Enforce dimensions strictly before generating embedding
            face_arr = face["face_array"]
            if face_arr.shape != (160, 160, 3):
                debug_log(f"Shape error: expected (160,160,3), got {face_arr.shape}")
                continue
                
            emb = generate_embedding(face_arr, facenet_model)
            debug_log("Embedding generated successfully.")
            
            name, confidence = recognize_face(emb, db)
            debug_log(f"Prediction completed: {name} ({confidence:.2f})")
            
            results.append((x, y, w, h, name, confidence))
        except Exception as e:
            debug_log(f"Error processing face: {e}")
            continue
    return results


    return results


# ─────────────────────────────────────────────────────────────────────────────
# WEBRTC PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.session_marked = set()
        self.unknown_logged = False
        self.detector = None
        self.facenet = None
        self.db = load_database()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Initialize models lazily inside the WebRTC thread to prevent Keras crash
        if self.detector is None:
            print("[DEBUG] Loading models in WebRTC thread...")
            self.detector, self.facenet = load_models()
            print("[DEBUG] Models loaded successfully.")
        
        # Phase 8: Process every 5th frame for extreme performance optimization (prevents freezing)
        if self.frame_count % 5 == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Performance Optimization: Resize frames before detection
            h, w = rgb.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                rgb_resized = cv2.resize(rgb, (640, int(h * scale)))
            else:
                rgb_resized = rgb

            # Phase 3: Unified preprocessing
            results = process_faces(rgb_resized, self.detector, self.facenet, self.db)
            
            # Scale bounding boxes back to original image size
            if scale != 1.0:
                scaled_results = []
                for (x, y, bw, bh, name, conf) in results:
                    scaled_results.append((
                        int(x / scale), int(y / scale),
                        int(bw / scale), int(bh / scale),
                        name, conf
                    ))
                self.last_results = scaled_results
            else:
                self.last_results = results
            
            for (x, y, w, h, name, conf) in self.last_results:
                if name != "Unknown":
                    if name not in self.session_marked:
                        if insert_attendance(name, "Present", conf):
                            self.session_marked.add(name)
                else:
                    if not self.unknown_logged:
                        insert_attendance("Unknown", "Unknown", conf)
                        self.unknown_logged = True

        if hasattr(self, 'last_results'):
            img = draw_results_on_frame(img, self.last_results)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    # Logo Handling
    logo_html = ""
    if logo_base64:
        # Properly displaying the logo if it exists
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="80" alt="Logo">'
    else:
        logo_html = '<div style="font-size: 3rem; margin-right: 20px;">🏛️</div>'

    st.markdown(f"""
    <div class="aror-header">
        {logo_html}
        <div>
            <h1>Smart Attendance System</h1>
            <p>Face Recognition Based Attendance • Aror University</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if logo_base64:
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="120"></div>', unsafe_allow_html=True)
        else:
            st.markdown("## 🏛️ Aror Univ")
        
        st.markdown("---")
        mode = st.radio(
            "Navigation",
            ["📷 Camera Mode", "🖼️ Upload Image", "📊 View Attendance"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        
        if st.button("🔄 Reload Database / Embeddings"):
            load_database.clear()
            st.success("✅ Database reloaded successfully!")
            
        st.markdown("### 📋 System Info")
        db = load_database()
        st.markdown(f"**Students Enrolled:** `{len(db)}`")
        st.markdown(f"**Threshold:** `{COSINE_SIMILARITY_THRESHOLD}`")
        
        return mode


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_metrics():
    summary = get_attendance_summary()
    total = summary['total_present'] + summary['total_unknown']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Present Today</div>
            <div class="metric-value" style="color: #ffffff;">{summary['total_present']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Unknown Access</div>
            <div class="metric-value" style="color: #aaaaaa;">{summary['total_unknown']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value" style="color: #dddddd;">{total}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────
def camera_mode():
    st.markdown("### 📷 Live Camera Recognition")
    st.info("💡 WebRTC Camera enabled. Make sure your browser allows camera access.")
    
    # Phase 1 & 6: Real-time stream with streamlit-webrtc.
    try:
        ctx = webrtc_streamer(
            key="attendance-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        if ctx.state.playing:
            st.success("✅ Camera active. Processing frames in real-time...")
    except Exception as e:
        st.error(f"❌ Error starting camera: {e}")

def upload_mode(detector, facenet, database):
    # Phase 2: Use st.file_uploader()
    st.markdown("### 🖼️ Batch Image Recognition")
    uploaded = st.file_uploader("Upload an image file (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        st.info("⏳ Processing Image...")
        
        try:
            # Phase 2: Convert uploaded file correctly using PIL -> numpy -> BGR
            image = Image.open(uploaded)
            img_rgb = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Phase 6: Invalid upload -> show error
            st.error(f"Invalid file format: {e}")
            return
        
        with st.spinner("Processing faces..."):
            # Phase 8: Resize frames before detection (Performance Optimization)
            h, w = img_rgb.shape[:2]
            scale = 1.0
            if w > 800:
                scale = 800.0 / w
                img_rgb_resized = cv2.resize(img_rgb, (800, int(h * scale)))
            else:
                img_rgb_resized = img_rgb

            # Phase 3: Unified preprocessing (process_faces applies same pipeline)
            results = process_faces(img_rgb_resized, detector, facenet, database, is_upload=True)
            
            # Scale boxes back
            if scale != 1.0:
                scaled_results = []
                for (x, y, bw, bh, name, conf) in results:
                    scaled_results.append((
                        int(x / scale), int(y / scale),
                        int(bw / scale), int(bh / scale),
                        name, conf
                    ))
                results = scaled_results
        
        # Phase 6: No face detected -> show warning
        if not results:
            st.warning("⚠️ No face detected in the uploaded image.")
            st.image(img_rgb, use_column_width=True)
            return
            
        if len(results) > 1:
            st.info(f"👥 Multiple faces detected: {len(results)}")
            
        st.success("✅ Image processed successfully.")
        
        annotated = draw_results_on_frame(img_bgr.copy(), results)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        for i, (x, y, w, h, name, conf) in enumerate(results):
            if name != "Unknown":
                sid, display = parse_student_display_name(name)
                insert_attendance(name, "Present", conf)
                st.markdown(f"""
                <div class="result-item">
                    <b>✅ {display}</b> (ID: {sid}) — Confidence: {conf:.4f}
                </div>
                """, unsafe_allow_html=True)
            else:
                insert_attendance("Unknown", "Unknown", conf)
                st.markdown(f"""
                <div class="result-item unknown">
                    <b>❌ Unknown Face Detected</b> — Confidence: {conf:.4f}
                </div>
                """, unsafe_allow_html=True)

def attendance_view():
    st.markdown("### 📊 Admin Panel: Attendance Records")
    
    # Simple Session State Authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.warning("🔒 This section is protected. Please enter the admin password.")
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            # Hardcoded password (1234)
            if pwd == "1234":
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # Authenticated Area
    if st.button("🔒 Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1:
        view = st.selectbox("Filter", ["Today", "All Time"])
    with col2:
        if st.button("🗑️ Clear Today's Log"):
            clear_today_attendance()
            st.rerun()

    records = get_today_attendance() if view == "Today" else get_all_attendance()
    
    if records:
        df = pd.DataFrame(records)
        # Prevent drop error if DataFrame is magically empty but records list isn't
        if not df.empty and "id" in df.columns:
            df = df.drop(columns=["id"])

        if not df.empty and "name" in df.columns:
            # Map names
            df["Display Name"] = df["name"].apply(lambda n: parse_student_display_name(n)[1] if n != "Unknown" else "Unknown")
            # Rename columns nicely
            df.rename(columns={"name": "Folder_ID", "status": "Status", "confidence": "Confidence", "date": "Date", "time": "Time"}, inplace=True)
            
            # Reorder
            df = df[["Display Name", "Folder_ID", "Status", "Confidence", "Date", "Time"]]

            def color_status(val):
                return 'color: #ffffff; font-weight: bold;' if val == 'Present' else 'color: #888888;'
            
            st.dataframe(df.style.map(color_status, subset=["Status"]), use_container_width=True)
    else:
        st.info("No records found.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_database()
    render_header()
    
    # Load Models & DB inside main
    detector, facenet = load_models()
    database = load_database()
    
    if not database:
        st.error("Database missing! Run `build_embeddings.py` first.")
        if st.button("🔄 Try Reloading Database"):
            load_database.clear()
            st.rerun()
        return

    mode = render_sidebar()
    render_metrics()
    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "📷 Camera Mode":
        camera_mode()
    elif mode == "🖼️ Upload Image":
        upload_mode(detector, facenet, database)
    elif mode == "📊 View Attendance":
        attendance_view()

if __name__ == "__main__":
    main()
