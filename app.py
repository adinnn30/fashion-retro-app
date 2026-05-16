import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- KONFIGURASI TAMPILAN RETRO CREAM-RED ---
st.set_page_config(page_title="Fashion AI Retro", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');

    .stApp {
        background-color: #F5F5DC; /* Cream Background */
    }
    
    h1, h2, h3, p, span, label {
        font-family: 'Special Elite', cursive;
        color: #8B0000; /* Retro Red */
    }

    .stFileUploader {
        border: 2px dashed #8B0000;
        background-color: #FFF8E7;
        padding: 10px;
    }

    .prediction-card {
        background-color: #8B0000;
        color: #F5F5DC;
        padding: 15px;
        border-radius: 10px;
        border: 4px double #F5F5DC;
        margin-bottom: 10px;
    }

    /* Mengubah warna progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF4500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # nama file sesuai dengan file .tflite
    interpreter = tf.lite.Interpreter(model_path="model_fashion.tflite")
    interpreter.allocate_tensors()
    return interpreter

# --- LABEL SESUAI DATA KAMU ---
CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear']

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing: Sesuaikan ukuran (misal 224x224 atau 150x150 sesuai modelmu)
    size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    
    # Normalisasi (jika saat training kamu bagi 255)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# --- UI UTAMA ---
st.markdown("<h1 style='text-align: center;'>FASHION CLASSIFIER v1.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload foto baju, sepatu, atau aksesoris untuk dicek AI</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Tampilkan Gambar dengan Frame Retro
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, use_column_width=True)
    
    st.markdown("<h3 style='text-align: center;'>HASIL ANALISIS:</h3>", unsafe_allow_html=True)
    
    with st.spinner('AI sedang berpikir...'):
        interpreter = load_model()
        result = predict(img, interpreter)
        
        # Urutkan dari yang paling tinggi probabilitasnya
        indices = np.argsort(result)[::-1]

        for i in indices:
            score = result[i] * 100
            label = CLASS_NAMES[i]
            
            # Tampilan hasil
            st.markdown(f"""
                <div class="prediction-card">
                    <span style='font-size: 20px;'>{label}</span> 
                    <span style='float: right;'>{score:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(score))

st.markdown("<br><br><p style='text-align: center; font-size: 12px;'>Kategori: Accessories (Jam, Tas, dll) | Apparel (Baju, Celana) | Footwear (Sepatu, Sandal)</p>", unsafe_allow_html=True)