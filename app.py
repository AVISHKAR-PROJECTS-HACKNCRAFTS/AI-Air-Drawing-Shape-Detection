import streamlit as st
import cv2
import numpy as np
from utils import HandTracker, preprocess_gesture, draw_perfect_shape, heuristic_classify
from model import load_model, SHAPES
import time

# Page Config
st.set_page_config(page_title="AI Air Drawing Shape Detector", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    }
    .stSidebar {
        background-color: rgba(45, 45, 68, 0.8);
        border-right: 1px solid #444;
    }
    h1 {
        color: #00f2fe;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00dbde 0%, #fc00ff 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(252, 0, 255, 0.4);
    }
    .shape-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        border-left: 5px solid #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 AI Air Canvas — Creative Shape Detector")
st.subheader("Draw & detect 12 creative shapes in real-time using CNN + Transformer AI")

# Show supported shapes
shape_emojis = {
    "Spiral": "🌀", "Infinity": "♾️", "Cloud": "☁️", "Lightning bolt": "⚡",
    "Flower": "🌸", "Butterfly": "🦋", "Crown": "👑", "Flame": "🔥",
    "Fish": "🐟", "Leaf": "🍃", "Music note": "🎵", "Smiley face": "😊"
}

# Initialize Session State
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'drawing' not in st.session_state:
    st.session_state.drawing = False
if 'points' not in st.session_state:
    st.session_state.points = []
if 'detected_shapes' not in st.session_state:
    st.session_state.detected_shapes = []
if 'current_color' not in st.session_state:
    st.session_state.current_color = (0, 255, 0) # Green

# Sidebar
st.sidebar.header("🎮 Controls")

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶️ Start Camera"):
    st.session_state.run_camera = True
if col_stop.button("⏹️ Stop Camera"):
    st.session_state.run_camera = False

color_picker = st.sidebar.color_picker("Pick a Shape Fill Color", "#00FF00")
st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("Use Heuristic Fallback", value=False, help="Uses geometric heuristics instead of the AI model. AI model is more advanced (CNN+Transformer).")

if st.sidebar.button("🚀 Re-Train AI Model"):
    with st.spinner("Generating data and training CNN+Transformer... This may take a few minutes."):
        from model import train_model
        train_model()
        st.cache_resource.clear()
        st.success("✅ Model trained and loaded!")
        st.rerun()

st.sidebar.markdown("---")
# Convert hex to RGB
hex_color = color_picker.lstrip('#')
st.session_state.current_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

if st.sidebar.button("🧹 Clear Canvas"):
    st.session_state.points = []
    st.session_state.detected_shapes = []

if st.sidebar.button("↩️ Undo Last Shape"):
    if st.session_state.detected_shapes:
        st.session_state.detected_shapes.pop()

st.sidebar.markdown("---")
st.sidebar.header("🎯 Supported Shapes")
shape_cols = st.sidebar.columns(2)
for i, shape in enumerate(SHAPES):
    emoji = shape_emojis.get(shape, "🔹")
    shape_cols[i % 2].markdown(f"{emoji} {shape}")

st.sidebar.markdown("---")
st.sidebar.header("📜 Detected Shapes")
for i, shape in enumerate(st.session_state.detected_shapes):
    emoji = shape_emojis.get(shape['name'], "🔹")
    st.sidebar.markdown(f"""
        <div class="shape-card">
            <strong>{emoji} {i+1}. {shape['name']}</strong>
        </div>
    """, unsafe_allow_html=True)

# AI Model
@st.cache_resource
def get_ai_model():
    return load_model()

model = get_ai_model()
tracker = HandTracker()

# Main Layout
col1, col2 = st.columns([3, 1])

with col1:
    st_frame = st.empty()

with col2:
    st.info("**How to use:**\n1. Start the camera\n2. Use your **index finger** to draw\n3. Keep other fingers closed\n4. When you stop moving or move away, AI classifies the shape\n\n**Supported shapes:**\n" + ", ".join([f"{shape_emojis.get(s, '')} {s}" for s in SHAPES]))
    prediction_label = st.empty()
    confidence_bar = st.empty()

# Camera Loop
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Hand Detection
        results = tracker.find_hand_landmarks(frame)
        cx, cy, landmarks = tracker.get_finger_tip(results, w, h)
        
        if landmarks:
            # Check gesture (e.g., if only index finger is up)
            st.session_state.points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 15, st.session_state.current_color, -1)
            cv2.putText(frame, "Drawing...", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tracker.mp_draw.draw_landmarks(frame, landmarks, tracker.mp_hands.HAND_CONNECTIONS)
        else:
            # If hand is gone and we have points, classify!
            if len(st.session_state.points) > 20:
                roi = preprocess_gesture(st.session_state.points)
                if roi is not None:
                    if demo_mode:
                        shape_name, confidence = heuristic_classify(st.session_state.points)
                    else:
                        # AI Classification
                        preds = model.predict(roi, verbose=0)
                        class_idx = np.argmax(preds[0])
                        confidence = float(np.max(preds[0]))
                        shape_name = SHAPES[class_idx]
                    
                    st.session_state.detected_shapes.append({
                        "name": shape_name,
                        "points": st.session_state.points.copy(),
                        "color": st.session_state.current_color
                    })
                    
                    emoji = shape_emojis.get(shape_name, "🔹")
                    prediction_label.success(f"{emoji} Detected: {shape_name}")
                    confidence_bar.progress(confidence)
                
                st.session_state.points = [] # Clear for next shape
        
        # Draw current points
        for i in range(1, len(st.session_state.points)):
            cv2.line(frame, st.session_state.points[i-1], st.session_state.points[i], st.session_state.current_color, 4)
            
        # Draw all previously detected shapes
        for shape in st.session_state.detected_shapes:
            frame = draw_perfect_shape(frame, shape['name'], shape['color'], shape['points'])
            
        # Display Frame
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
        # Small sleep to reduce CPU usage
        time.sleep(0.01)
        
    cap.release()
else:
    st.write("Camera is off. Click '▶️ Start Camera' in the sidebar to begin.")
