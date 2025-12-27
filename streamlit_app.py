import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import zipfile
import os
import gdown

# =============================================
# CẤU HÌNH MODEL
# =============================================
MODEL_FILE = "lung_4_classes_fixed.keras"      # Tên file keras sau khi giải nén/tải về
ZIP_FILE = "lung_4_classes_model.zip"         # Tên file zip (nếu bạn muốn thêm zip vào repo sau này)
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1LpZeK3Em1hDxNd4rXzhgm9huvdGuakAr"

# =============================================
# TỰ ĐỘNG LẤY MODEL (theo thứ tự ưu tiên)
# =============================================
if not os.path.exists(MODEL_FILE):
    # Ưu tiên 1: Có file zip trong repo → giải nén
    if os.path.exists(ZIP_FILE):
        st.info("🔄 Đang giải nén model từ file zip trong repo...")
        with st.spinner("Giải nén..."):
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".")
        st.success("✅ Giải nén thành công!")

    # Ưu tiên 2: Không có zip → tải từ Google Drive
    else:
        st.info("🌐 Đang tải model từ Google Drive (~53MB). Lần đầu có thể mất vài phút...")
        with st.spinner("Đang tải model..."):
            gdown.download(DRIVE_URL, MODEL_FILE, quiet=False)
        st.success("✅ Tải model thành công!")

else:
    st.info("✅ Model đã có sẵn trên máy.")

# =============================================
# LOAD MODEL (chỉ load 1 lần)
# =============================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

with st.spinner("Đang load model vào bộ nhớ..."):
    model = load_model()

# =============================================
# CẤU HÌNH GIAO DIỆN
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

# =============================================
# UPLOAD VÀ DỰ ĐOÁN
# =============================================
uploaded_file = st.file_uploader("Upload ảnh X-quang (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã upload", width=400)

    with st.spinner("Đang phân tích ảnh..."):
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100

    st.markdown("---")
    predicted_name = class_names[predicted_idx]

    # Hiển thị kết quả chính
    if predicted_name == 'Phổi bình thường (Normal)':
        st.success(f"**Kết quả: {predicted_name}** (Không phát hiện dấu hiệu bất thường)")
    elif confidence >= 70:
        st.success(f"**Kết quả: {predicted_name}**")
    elif confidence >= 50:
        st.warning(f"**Kết quả: {predicted_name}** (Độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {predicted_name}** (Độ tin cậy thấp)")

    st.write(f"**Độ tin cậy cao nhất: {confidence:.2f}%**")

    # Chi tiết xác suất từng lớp
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
    st.markdown("### Hướng dẫn:")
    st.write("- Ảnh nên là X-quang ngực thẳng (PA hoặc AP)")
    st.write("- Định dạng: JPG, PNG, JPEG")
    st.write("- Kết quả chỉ mang tính tham khảo")
