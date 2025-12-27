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
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1LpZeK3Em1hDxNd4rXzhgm9huvdGuakAr"  # Link direct chuẩn

# =============================================
# TỰ ĐỘNG TẢI MODEL (chỉ từ Google Drive)
# =============================================
if not os.path.exists(MODEL_FILE):
    st.info("🌐 Đang tải model từ Google Drive (~53MB). Lần đầu sẽ mất 3-7 phút, vui lòng chờ...")
    with st.spinner("Đang tải file lớn (bypass xác thực Google)..."):
        # Xóa file cũ nếu bị hỏng từ lần trước
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        # Tải với fuzzy=True để xử lý trang confirm virus scan
        gdown.download(DRIVE_URL, MODEL_FILE, quiet=False, fuzzy=True)
    st.success("✅ Tải model thành công!")

else:
    st.info("✅ Model đã có sẵn trên server.")

# =============================================
# LOAD MODEL (KHÔNG DÙNG CACHE để tránh lỗi cache hỏng)
# =============================================
st.write("🔄 Đang load model vào bộ nhớ... (có thể mất 30-90 giây)")
with st.spinner("Loading Keras model..."):
    model = tf.keras.models.load_model(MODEL_FILE)

st.success("✅ Model đã load thành công và sẵn sàng dự đoán!")

# =============================================
# CẤU HÌNH GIAO DIỆN (giữ nguyên phần cũ của bạn)
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
# UPLOAD VÀ DỰ ĐOÁN (giữ nguyên code cũ của bạn)
# =============================================
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
    st.markdown("### Hướng dẫn:")
    st.write("- Ảnh nên là X-quang ngực thẳng (PA hoặc AP)")
    st.write("- Định dạng: JPG, PNG, JPEG")
    st.write("- Kết quả chỉ mang tính tham khảo")
