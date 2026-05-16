import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- KONFIGURASI TAMPILAN PASTEL IMUT ---
st.set_page_config(page_title="Fashion Kawaii AI", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&display=swap');

    /* Latar Belakang Pastel Pink Lembut */
    .stApp {
        background-color: #FFF0F5 !important; 
    }
    
    /* Font Bulat & Lucu */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, span, label {
        font-family: 'Quicksand', sans-serif !important;
        color: #5D5D5D !important; /* Abu-abu gelap supaya jelas dibaca */
    }

    /* Judul Pelangi */
    .judul-imut {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        background: -webkit-linear-gradient(#FFB6C1, #87CEFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* Kartu Hasil Warna-Warni Pastel */
    .card {
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .card-accessories { background-color: #FFD1DC !important; } /* Pastel Pink */
    .card-apparel { background-color: #C1E1C1 !important; }     /* Pastel Green */
    .card-footwear { background-color: #B2CEFE !important; }    /* Pastel Blue */

    /* Tulisan di dalam kartu */
    .card-text-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #4A4A4A !important;
    }
    .card-text-score {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Tombol Cantik */
    .stButton>button {
        background: linear-gradient(45deg, #FFB6C1, #FFD1DC) !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 182, 193, 0.4);
    }

    /* Progress bar warna soft pink */
    .stProgress > div > div > div > div {
        background-color: #FFB6C1 !important;
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

# Mapping Class
CLASS_INFO = [
    {"name": "ACCESSORIES 💍", "class": "card-accessories"},
    {"name": "APPAREL 👕", "class": "card-apparel"},
    {"name": "FOOTWEAR 👟", "class": "card-footwear"}
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

# --- UI ---
st.markdown("<h1 class='judul-imut'>✨ Fashion Kawaii AI ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Yuk, cek kategori fashion kamu di sini! 💕</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Tampilkan gambar dengan border bulat imut
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button('🌸 Tebak Sekarang! 🌸'):
        interpreter = load_model()
        if interpreter:
            with st.spinner('Tunggu sebentar ya, cantik... ✨'):
                result = predict(img, interpreter)
                indices = np.argsort(result)[::-1]

                st.markdown("<h3 style='text-align: center; margin-top:20px;'>Hore! Ini Hasilnya:</h3>", unsafe_allow_html=True)
                
                for i in indices:
                    score = result[i] * 100
                    info = CLASS_INFO[i]
                    
                    # Tampilan kartu warna-warni
                    st.markdown(f"""
                        <div class="card {info['class']}">
                            <div class="card-text-title">{info['name']}</div>
                            <div class="card-text-score">{score:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(score))

st.markdown("<br><p style='text-align: center; font-size: 14px;'>Dibuat dengan ❤️ untuk fashionista!</p>", unsafe_allow_html=True)