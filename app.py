import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- TAMPILAN KATALOG RETRO (RED ON CREAM) ---
st.set_page_config(page_title="Fashion AI Katalog", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

    /* Latar Belakang Cream */
    .stApp {
        background-color: #FDF5E6 !important; 
    }
    
    /* Font Semua Tulisan Jadi Typewriter Style Merah */
    section, div, p, h1, h2, h3, span, label {
        font-family: 'Courier Prime', monospace !important;
        color: #B22222 !important; /* Merah Marun Retro */
    }

    /* Garis Pembatas Tebal ala Koran Lama */
    .retro-border {
        border: 3px solid #B22222;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #FDF5E6;
    }

    /* Judul Besar */
    .judul-katalog {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 8px double #B22222;
        margin-bottom: 20px;
    }

    /* Progress Bar Merah */
    .stProgress > div > div > div > div {
        background-color: #B22222 !important;
    }

    /* Tombol Retro */
    .stButton>button {
        background-color: #B22222 !important;
        color: #FDF5E6 !important;
        border-radius: 0px !important;
        border: none !important;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
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

CLASS_NAMES = ['ACCESSORIES', 'APPAREL', 'FOOTWEAR']

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

# --- UI ---
st.markdown("<div class='judul-katalog'>FASHION AI<br>VINTAGE EDITION</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("UPLOAD FOTO DISINI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Frame foto ala koran
    st.markdown("<div style='border: 2px solid #B22222; padding: 10px;'>", unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button('KLASIFIKASI SEKARANG'):
        interpreter = load_model()
        if interpreter:
            with st.spinner('MENGANALISIS...'):
                result = predict(img, interpreter)
                indices = np.argsort(result)[::-1]

                st.markdown("<h2 style='text-align: center; border-top: 2px solid #B22222; margin-top:20px;'>HASIL PREDIKSI</h2>", unsafe_allow_html=True)
                
                for i in indices:
                    score = result[i] * 100
                    # Tampilan teks merah di atas cream
                    st.markdown(f"""
                        <div class="retro-border">
                            <span style='font-size: 20px; font-weight: bold;'>{CLASS_NAMES[i]}</span>
                            <span style='float: right; font-size: 25px; font-weight: bold;'>{score:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(score))

st.markdown("<br><p style='text-align: center; border-top: 1px solid #B22222;'>© 2024 FASHION AI CATALOG - ALL RIGHTS RESERVED</p>", unsafe_allow_html=True)