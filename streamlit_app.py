import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# =============================================
# CẤU HÌNH (dùng /tmp để tránh vấn đề disk)
# =============================================
MODEL_PATH = "/tmp/lung_4_classes_fixed.keras"  # Lưu tạm trên server
DRIVE_ID = "1LpZeK3Em1hDxNd4rXzhgm9huvdGuakAr"   # Chỉ ID cho ngắn
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

# =============================================
# TẢI & CACHE MODEL (cách tốt nhất cho Streamlit Cloud)
# =============================================
@st.cache_resource(show_spinner="Đang tải và chuẩn bị model (~53MB, lần đầu mất 3-10 phút)...")
def download_and_load_model():
    # Tải với fuzzy=True để bypass confirm
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
    
    # Load model ngay sau tải
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Gọi hàm (sẽ cache toàn bộ: tải chỉ 1 lần, load chỉ 1 lần)
try:
    model = download_and_load_model()
    st.success("✅ Model đã sẵn sàng! Bạn có thể upload ảnh ngay.")
except Exception as e:
    st.error("Lỗi tải/load model. Chi tiết (cho debug): " + str(e))
    st.stop()

# =============================================
# GIAO DIỆN & DỰ ĐOÁN (giữ nguyên đẹp như cũ)
# =============================================
class_names = ['COVID-19', 'Phổi bình thường (Normal)', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

st.set_page_config(page_title="AI Phân loại X-quang Phổi (4 lớp)", layout="centered")
st.title("🫁 AI Nhận diện 4 bệnh phổi từ ảnh X-quang")
st.markdown("---")
st.write("""
**Phân loại 4 lớp:**
- COVID-19
- Phổi bình thường (Normal)
- Viêm phổi (Pneumonia)
- Lao phổi (Tuberculosis)
""")
st.error("⚠️ **Kết quả chỉ mang tính tham khảo – Không thay thế chẩn đoán của bác sĩ!**")

uploaded_file = st.file_uploader("Upload ảnh X-quang (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã upload", width=400)

    with st.spinner("Đang phân tích ảnh..."):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100

    st.markdown("---")
    predicted_name = class_names[predicted_idx]

    if predicted_name == 'Phổi bình thường (Normal)':
        st.success(f"**Kết quả: {predicted_name}** (Không phát hiện dấu hiệu bất thường)")
    elif confidence >= 70:
        st.success(f"**Kết quả: {predicted_name}**")
    elif confidence >= 50:
        st.warning(f"**Kết quả: {predicted_name}** (Độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {predicted_name}** (Độ tin cậy thấp)")

    st.write(f"**Độ tin cậy cao nhất: {confidence:.2f}%**")

    st.markdown("### Xác suất chi tiết từng lớp:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        progress_val = max(0.0, min(1.0, prob / 100))
        st.progress(progress_val)
        if i == predicted_idx:
            st.write(f"**{name}: {prob:.2f}%** 👈")
        else:
            st.write(f"{name}: {prob:.2f}%")

    st.info("💡 **Khuyến nghị**: Hãy mang kết quả này đến bác sĩ để được tư vấn chính xác!")

else:
    st.info("👆 Vui lòng upload ảnh X-quang để bắt đầu phân tích.")
