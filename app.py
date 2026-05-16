import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# Trik: Import tflite-runtime dengan aman
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

# --- TAMPILAN RETRO ---
st.set_page_config(page_title="Fashion Retro AI", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');
    
    .stApp {
        background-color: #F5F5DC; /* Cream */
    }
    
    h1, h2, h3, p, span, label {
        font-family: 'Special Elite', cursive;
        color: #8B0000; /* Retro Red */
    }

    .stFileUploader {
        border: 3px dashed #8B0000 !important;
        background-color: #FFFDD0;
    }

    .prediction-card {
        background-color: #8B0000;
        color: #F5F5DC;
        padding: 20px;
        border-radius: 0px;
        border: 5px double #F5F5DC;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .stProgress > div > div > div > div {
        background-color: #FF4500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_fashion.tflite")
    interpreter.allocate_tensors()
    return interpreter

# SESUAIKAN DENGAN KELAS KAMU
CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear']

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape from model
    input_shape = input_details[0]['shape']
    size = (input_shape[1], input_shape[2])
    
    # Preprocess image
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    
    # Normalisasi (Jika saat training dibagi 255)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# --- UI UTAMA ---
st.markdown("<h1 style='text-align: center; border-bottom: 5px solid #8B0000;'>FASHION RETRO ANALYZER</h1>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("MASUKKAN GAMBAR FASHION (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Menampilkan Gambar
    st.image(img, caption="Foto Yang Diupload", use_container_width=True)
    
    if st.button('MULAI ANALISIS AI'):
        with st.spinner('Mesin Retro Sedang Berputar...'):
            try:
                interpreter = load_model()
                result = predict(img, interpreter)
                
                # Urutkan hasil
                indices = np.argsort(result)[::-1]

                st.markdown("<h3 style='text-align: center;'>HASIL PREDIKSI:</h3>", unsafe_allow_html=True)
                
                for i in indices:
                    score = result[i] * 100
                    label = CLASS_NAMES[i]
                    
                    # Tampilan kartu hasil
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h2 style='margin:0;'>{label}</h2>
                            <h1 style='margin:0; color: #FFF;'>{score:.1f}%</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(score))
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

st.markdown("<br><hr><p style='text-align: center;'>Kategori: Accessories (Jam, Tas) | Apparel (Baju) | Footwear (Sepatu)</p>", unsafe_allow_html=True)