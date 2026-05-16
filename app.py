import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps
import numpy as np

# --- KONFIGURASI TAMPILAN ---
st.set_page_config(page_title="Fashion AI Retro", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');
    .stApp { background-color: #F5F5DC; }
    h1, h2, h3, p, span, label { font-family: 'Special Elite', cursive; color: #8B0000; }
    .stFileUploader { border: 2px dashed #8B0000; background-color: #FFF8E7; }
    .prediction-card {
        background-color: #8B0000; color: #F5F5DC;
        padding: 15px; border-radius: 10px;
        border: 4px double #F5F5DC; margin-bottom: 10px;
    }
    .stProgress > div > div > div > div { background-color: #FF4500; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (Pake TFLite Runtime) ---
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_fashion.tflite")
    interpreter.allocate_tensors()
    return interpreter

CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear']

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize otomatis ngikutin model
    size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    
    # Normalisasi
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# --- UI ---
st.markdown("<h1 style='text-align: center;'>FASHION CLASSIFIER v1.0</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    if st.button('TEBAK SEKARANG'):
        with st.spinner('Analisis...'):
            interpreter = load_model()
            result = predict(img, interpreter)
            indices = np.argsort(result)[::-1]

            for i in indices:
                score = result[i] * 100
                st.markdown(f"""
                    <div class="prediction-card">
                        <span style='font-size: 20px;'>{CLASS_NAMES[i]}</span> 
                        <span style='float: right;'>{score:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))