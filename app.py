import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Fashion Candy AI", layout="centered")

# --- CSS CUSTOM (PASTEL LUXURY STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Varela+Round&display=swap');

    /* Background Utama */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%) !important;
    }

    /* Font Global */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, span, label {
        font-family: 'Varela Round', sans-serif !important;
        color: #444444 !important;
    }

    /* Header Section */
    .main-header {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }

    /* Judul Style */
    .judul-text {
        font-size: 38px;
        font-weight: bold;
        background: -webkit-linear-gradient(#FF9A9E, #FAD0C4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Kartu Info Kategori di Depan */
    .info-container {
        display: flex;
        gap: 10px;
        margin-bottom: 30px;
        justify-content: center;
        flex-wrap: wrap;
    }
    .info-box {
        background: white;
        padding: 15px;
        border-radius: 20px;
        width: 100%;
        max-width: 200px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border-bottom: 5px solid #eee;
    }

    /* Kartu Hasil Prediksi */
    .res-card {
        background: white;
        padding: 25px;
        border-radius: 25px;
        margin-top: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border-left: 10px solid #eee;
    }
    .res-title { font-size: 18px; font-weight: bold; color: #666; margin-bottom: 5px; }
    .res-score { font-size: 40px; font-weight: 800; margin: 0; }

    /* Warna per Kategori */
    .border-accessories { border-left-color: #FFB7B2 !important; color: #FFB7B2 !important; }
    .border-apparel { border-left-color: #B2E2F2 !important; color: #B2E2F2 !important; }
    .border-footwear { border-left-color: #B2F2BB !important; color: #B2F2BB !important; }

    /* Button */
    .stButton>button {
        background: linear-gradient(to right, #FF9A9E 0%, #FAD0C4 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        border-radius: 50px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        width: 100% !important;
        margin-top: 20px;
    }

    /* Uploader */
    .stFileUploader {
        border: 2px dashed #FF9A9E !important;
        border-radius: 20px !important;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "model_fashion.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

# Konfigurasi Kategori
CATS = [
    {"id": "Accessories", "name": "ACCESSORIES", "emoji": "💍", "desc": "Jam, Tas, Kacamata, Perhiasan", "cls": "border-accessories"},
    {"id": "Apparel", "name": "APPAREL", "emoji": "👕", "desc": "Baju, Celana, Jaket, Rok", "cls": "border-apparel"},
    {"id": "Footwear", "name": "FOOTWEAR", "emoji": "👟", "desc": "Sepatu, Sandal, Wedges", "cls": "border-footwear"}
]

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- HALAMAN DEPAN ---
st.markdown("""
    <div class="main-header">
        <div class="judul-text">Fashion Candy AI</div>
        <p style="margin:0; color:#888;">Klasifikasi Produk Fashion secara Ajaib ✨</p>
    </div>
    """, unsafe_allow_html=True)

# Guide Kategori
st.markdown("<p style='text-align:center; font-weight:bold; margin-bottom:15px;'>Kategori yang dapat dideteksi:</p>", unsafe_allow_html=True)
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown(f"<div class='info-box' style='border-color:#FFB7B2'><h3>💍</h3><b>Accessories</b><br><small>{CATS[0]['desc']}</small></div>", unsafe_allow_html=True)
with col_info2:
    st.markdown(f"<div class='info-box' style='border-color:#B2E2F2'><h3>👕</h3><b>Apparel</b><br><small>{CATS[1]['desc']}</small></div>", unsafe_allow_html=True)
with col_info3:
    st.markdown(f"<div class='info-box' style='border-color:#B2F2BB'><h3>👟</h3><b>Footwear</b><br><small>{CATS[2]['desc']}</small></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("Upload Foto Fashion Kamu Disini", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preview Image
    st.image(img, use_container_width=True)
    
    if st.button('✨ Mulai Tebak Sekarang! ✨'):
        interpreter = load_model()
        if interpreter:
            with st.spinner('AI sedang melihat fotomu... 🍦'):
                result = predict(img, interpreter)
                indices = np.argsort(result)[::-1]

                st.markdown("<h3 style='text-align: center; margin-top:30px;'>Hasil Tebakan AI:</h3>", unsafe_allow_html=True)
                
                for i in indices:
                    score = result[i] * 100
                    cat = CATS[i]
                    
                    st.markdown(f"""
                        <div class="res-card {cat['cls']}">
                            <p class="res-title">{cat['emoji']} {cat['name']}</p>
                            <p class="res-score" style="color: inherit;">{score:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(score))

st.markdown("<br><br><p style='text-align:center; color:#bbb; font-size:12px;'>© 2024 Fashion Candy Classifier</p>", unsafe_allow_html=True)