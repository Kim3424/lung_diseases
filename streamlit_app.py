import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# =============================================
# CẤU HÌNH MODEL
# =============================================
MODEL_FILE = "lung_4_classes_fixed.keras"
DRIVE_URL = "https://drive.google.com/file/d/1LpZeK3Em1hDxNd4rXzhgm9huvdGuakAr/view?usp=drive_link"

# =============================================
# TỰ ĐỘNG TẢI MODEL (chỉ từ Google Drive)
# =============================================
if not os.path.exists(MODEL_FILE):
    st.info("🌐 Đang tải model từ Google Drive (~53MB). Lần đầu sẽ mất 2-5 phút...")
    with st.spinner("Đang tải và xác thực file..."):
        # Xóa file cũ nếu bị hỏng
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        # Tải lại sạch
        gdown.download(DRIVE_URL, MODEL_FILE, quiet=False)
    st.success("✅ Tải model thành công!")

else:
    st.info("✅ Model đã có sẵn.")

# =============================================
# LOAD MODEL (KHÔNG DÙNG CACHE để tránh BadZipFile)
# =============================================
st.write("🔄 Đang load model vào bộ nhớ... (có thể mất 30-60 giây)")
with st.spinner("Loading model..."):
    model = tf.keras.models.load_model(MODEL_FILE)

st.success("✅ Model đã load thành công và sẵn sàng dự đoán!")
