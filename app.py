import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- TAMPILAN RETRO ---
st.set_page_config(page_title="Fashion Retro AI", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');
    .stApp { background-color: #F5F5DC; }
    h1, h2, h3, p, span, label { font-family: 'Special Elite', cursive; color: #8B0000; }
    .prediction-card {
        background-color: #8B0000; color: #F5F5DC;
        padding: 15px; border-radius: 0px;
        border: 5px double #F5F5DC; margin-bottom: 10px;
        text-align: center;
    }
    .stProgress > div > div > div > div { background-color: #FF4500; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "model_fashion.tflite"
    
    # Cek apakah file benar-benar ada
    if not os.path.exists(model_path):
        st.error(f"File {model_path} tidak ditemukan di folder utama!")
        return None
        
    try:
        # Menggunakan tf.lite langsung dari tensorflow-cpu
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear']

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape
    input_shape = input_details[0]['shape']
    size = (input_shape[1], input_shape[2])
    
    # Preprocess
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    
    # Normalisasi (Samakan dengan training kamu)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# --- UI ---
st.markdown("<h1 style='text-align: center;'>FASHION RETRO AI</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("UPLOAD FOTO FASHION", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    if st.button('TEBAK KATEGORI'):
        interpreter = load_model()
        
        if interpreter:
            with st.spinner('Sedang Berpikir...'):
                try:
                    result = predict(img, interpreter)
                    indices = np.argsort(result)[::-1]

                    st.markdown("### HASIL ANALISIS:")
                    for i in indices:
                        score = result[i] * 100
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h3 style='margin:0;'>{CLASS_NAMES[i]}</h3>
                                <h2 style='margin:0;'>{score:.1f}%</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(score))
                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")