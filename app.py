import streamlit as st
import os
import pickle
import numpy as np
import cv2
import pandas as pd
from deepface import DeepFace
from PIL import Image

# --- 1. PAGE SETUP & UI STYLING ---
st.set_page_config(page_title="Lazy Amaan's Face Finder", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500;600;800&display=swap');
    
    /* Force Streamlit fonts */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    
    /* ── Animated scanline grain overlay ── */
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none; z-index: 9999; opacity: 0.4;
    }

    /* ── Ambient glow blobs ── */
    .blob {
        position: fixed; border-radius: 50%;
        filter: blur(120px); pointer-events: none; z-index: 0;
        animation: drift 12s ease-in-out infinite alternate;
    }
    .blob1 { width:500px; height:500px; background: rgba(123,97,255,0.07); top:-100px; left:-80px; }
    .blob2 { width:400px; height:400px; background: rgba(0,229,255,0.06); bottom:0; right:-60px; animation-delay:-6s; }
    .blob3 { width:300px; height:300px; background: rgba(255,77,109,0.04); top:40%; left:40%; animation-delay:-3s; }

    @keyframes drift {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(40px, 30px) scale(1.06); }
    }
    
    /* ── Text Styling ── */
    .hero-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4.5rem; 
        background: linear-gradient(90deg, #00e5ff, #7b61ff 40%, #ff4d6d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        line-height: 1.05;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: -2px;
    }
    
    .sarcastic-subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    .tech-pill {
        display: inline-block;
        background: rgba(0,229,255,0.07);
        border: 1px solid rgba(0,229,255,0.22);
        color: #00e5ff;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 20px;
        margin-right: 10px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,229,255,0.1);
    }
    
    /* Overriding Streamlit's default button to look like your action-btn */
    .stButton > button {
        background: linear-gradient(135deg, #7b61ff 0%, #00e5ff 100%);
        color: #000 !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        border: none;
        transition: transform 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,229,255,0.4);
        border: none;
    }
</style>

<div class="blob blob1"></div>
<div class="blob blob2"></div>
<div class="blob blob3"></div>
""", unsafe_allow_html=True)

# --- 2. BACKEND CONFIGURATION ---
DB_FILE = "master_fest_database.pkl"
MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
ONEDRIVE_URL = "https://mnnitedu-my.sharepoint.com/personal/devadathan_20253548_mnnit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdevadathan%5F20253548%5Fmnnit%5Fac%5Fin%2FDocuments%2FCulrav%2025&ga=1"

@st.cache_data 
def load_database():
    if not os.path.exists(DB_FILE):
        return None
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)

def extract_faces_from_uploads(uploaded_files):
    embeddings = []
    for file in uploaded_files:
        try:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            results = DeepFace.represent(img_path=img, model_name=MODEL_NAME, 
                                         detector_backend=DETECTOR, enforce_detection=True)
            embeddings.append(np.array(results[0]["embedding"]))
        except Exception as e:
            st.error(f"⛔ My AI couldn't find a face in {file.name}. Are you sure that's a selfie? Error: {e}")
            return None
    return embeddings

# --- 3. SIDEBAR (USER CONTROLS) ---
with st.sidebar:
    st.markdown("<h2 style='font-family: Share Tech Mono; color: #00e5ff; font-size: 2.5rem;'>◈ FaceFind</h2>", unsafe_allow_html=True)
    st.markdown("<h3>   BECAUSE SCROLLING IS FOR PEASANTS 😋</h4>" , unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📸 Teach the AI your face")
    uploaded_selfies = st.file_uploader("Drop 1-3 selfies here. Make sure you actually look good in them.", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    st.markdown("### 🎯 Search Strictness")
    top_n = st.slider("How many photos do you want?", min_value=5, max_value=100, value=30)
    threshold = st.slider("AI Strictness (Recommended : 0.6 - 0.65)", min_value=0.40, max_value=0.75, value=0.60, step=0.01)
    
    search_clicked = st.button("⚡ FIND MY FACE ALREADY", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("Sarcasrically Engineered by **Amaan Arif 🫶🏻** ")

# --- 4. MAIN STAGE ---
st.markdown("<div class='hero-title'>Media House of Mnnit(MHM) <br>Leftover Work</div>", unsafe_allow_html=True)
st.markdown("<div class='sarcastic-subtitle'>Upload a selfie and let my heavily over-engineered AI scan 10,000+ faces.<br>Stop looking like a background NPC and go find your photos.</div>", unsafe_allow_html=True)

# The OneDrive Link Button
st.link_button("☁️ Open the Massive Culrav '25 OneDrive", ONEDRIVE_URL, type="secondary")

st.markdown("""
    <span class='tech-pill'>⚡ ArcFace 512-d Vector</span>
    <span class='tech-pill'>⚡ MTCNN Vision</span>
    <span class='tech-pill'>⚡ Zero Patience</span>
""", unsafe_allow_html=True)
st.divider()

# --- 5. EXECUTION ENGINE ---
if search_clicked:
    if not uploaded_selfies:
        st.warning("⚠ Bro, you have to upload a selfie first. The AI isn't telepathic.")
    else:
        database = load_database()
        if database is None:
            st.error(f"⛔ The database is missing! Did you forget to run the 80-minute scan, Amaan?")
        else:
            with st.spinner("🧠 Judging your facial geometry..."):
                selfie_vectors = extract_faces_from_uploads(uploaded_selfies)
            
            if selfie_vectors:
                with st.spinner(f"🔍 Hunting down your face among {len(database)} people..."):
                    matches = []
                    for stored_face in database:
                        stored_vector = np.array(stored_face["embedding"])
                        best_distance = 1.0 
                        
                        for selfie_vector in selfie_vectors:
                            distance = np.dot(selfie_vector, stored_vector) / (np.linalg.norm(selfie_vector) * np.linalg.norm(stored_vector))
                            distance = 1 - distance 
                            if distance < best_distance:
                                best_distance = distance
                        
                        if best_distance <= threshold:
                            matches.append({
                                "path": stored_face["file_path"],
                                "distance": best_distance
                            })

                    matches = sorted(matches, key=lambda x: x["distance"])
                    unique_matches, seen_paths = [], set()
                    for match in matches:
                        if match["path"] not in seen_paths and os.path.exists(match["path"]):
                            unique_matches.append(match)
                            seen_paths.add(match["path"])

                if not unique_matches:
                    st.info("💀 No matches found. Either the AI is blind, or you really didn't leave your hostel room during the fest. (Try sliding the strictness bar to the right!)")
                else:
                    st.success(f"🏆 Found {len(unique_matches)} photos where you don't look terrible!")
                    
                    # 5A. Create Image Grid Gallery
                    cols = st.columns(4)
                    for i in range(min(top_n, len(unique_matches))):
                        match = unique_matches[i]
                        col_idx = i % 4
                        
                        with cols[col_idx]:
                            img = Image.open(match["path"])
                            img.thumbnail((600, 600)) 
                            st.image(img, use_container_width=True)
                            folder_name = os.path.basename(os.path.dirname(match["path"]))
                            score_percent = round((1 - match['distance']) * 100, 1)
                            st.caption(f"📂 {folder_name} | Match: {score_percent}%")
                    
                    st.divider()
                    
                    # 5B. Generate the Searchable Data Table
                    st.markdown("### 📋 Copy these file names. Do the rest yourself.")
                    st.markdown("I already found the photos. The least you can do is copy the file name below, click the OneDrive button at the top, and paste it in the search bar to get the HD version.")
                    
                    table_data = []
                    for i in range(min(top_n, len(unique_matches))):
                        match = unique_matches[i]
                        file_name = os.path.basename(match["path"])
                        folder_name = os.path.basename(os.path.dirname(match["path"]))
                        score_percent = round((1 - match['distance']) * 100, 1)
                        
                        table_data.append({
                            "Event / Folder": folder_name,
                            "File Name": file_name,
                            "AI Confidence": f"{score_percent}%"
                        })
                        
                    # Convert to Pandas DataFrame and display as an interactive table
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Drop a selfie in the sidebar and click the button. I'm not scrolling through that drive for you.")