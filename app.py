import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- KONFIGURASI TAMPILAN RETRO MODERN (CREAM-RED) ---
st.set_page_config(page_title="Fashion Retro AI", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');

    /* Warna Latar Belakang Seluruh Web (Cream) */
    .stApp {
        background-color: #F5F5DC; 
    }
    
    /* Warna Tulisan Umum (Merah Gelap) */
    h1, h2, h3, p, span, label {
        font-family: 'Special Elite', cursive;
        color: #8B0000 !important;
    }

    /* Kotak Hasil Prediksi (Merah) */
    .prediction-card {
        background-color: #8B0000; /* Latar Kotak Merah */
        padding: 20px;
        border-radius: 0px;
        border: 5px double #F5F5DC; /* Frame Cream */
        margin-bottom: 10px;
        text-align: center;
        box-shadow: 5px 5px 0px #5a0000;
    }

    /* TULISAN DI DALAM KOTAK PREDIKSI (CREAM) */
    .prediction-card h2, .prediction-card h3 {
        color: #F5F5DC !important; /* Tulisan jadi Cream di atas Merah */
        margin: 0;
        font-family: 'Special Elite', cursive;
    }

    /* Garis pemisah retro */
    .retro-line {
        border-top: 5px solid #8B0000;
        margin: 20px 0;
    }

    /* Progress bar warna Oranye Retro */
    .stProgress > div > div > div > div {
        background-color: #FF4500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "model_fashion.tflite"
    if not os.path.exists(model_path):
        st.error(f"File {model_path} tidak ditemukan!")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

# Urutan Class: 0: Accessories, 1: Apparel, 2: Footwear
CLASS_NAMES = ['ACCESSORIES', 'APPAREL', 'FOOTWEAR']

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessing
    input_shape = input_details[0]['shape']
    size = (input_shape[1], input_shape[2])
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- TAMPILAN UTAMA ---
st.markdown("<h1 style='text-align: center;'>🧥 FASHION RETRO ANALYZER 🧥</h1>", unsafe_allow_html=True)
st.markdown("<div class='retro-line'></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("MASUKKAN FOTO (Baju/Sepatu/Jam)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Tampilkan Gambar
    st.image(img, use_container_width=True)
    
    if st.button('TEBAK SEKARANG'):
        interpreter = load_model()
        if interpreter:
            with st.spinner('AI SEDANG BERPIKIR...'):
                result = predict(img, interpreter)
                indices = np.argsort(result)[::-1]

                st.markdown("<h3 style='text-align: center;'>HASIL ANALISIS MESIN:</h3>", unsafe_allow_html=True)
                
                for i in indices:
                    score = result[i] * 100
                    # Tampilan kotak Merah dengan Tulisan Cream
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3>{CLASS_NAMES[i]}</h3>
                            <h2>{score:.1f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(score))

st.markdown("<br><div class='retro-line'></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Kategori: Accessories (Jam, Tas, Kacamata) | Apparel (Baju, Celana) | Footwear (Sepatu, Sandal)</p>", unsafe_allow_html=True)