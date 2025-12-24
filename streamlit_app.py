import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import zipfile
import os

# =============================================
# GIẢI NÉN MÔ HÌNH (nếu dùng zip)
# =============================================
model_file = "lung_4_classes_model.keras"
zip_file = "lung_4_classes_model.zip"

if not os.path.exists(model_file):
    if os.path.exists(zip_file):
        st.write("🔄 Giải nén mô hình 4 lớp...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Giải nén OK!")
    else:
        st.error(f"❌ Không tìm thấy file: {model_file} hoặc {zip_file}")
        st.stop()
else:
    st.write("✅ Mô hình đã sẵn sàng.")

# Load model
with st.spinner("Đang load mô hình 4 lớp..."):
    model = tf.keras.models.load_model(model_file)

# Thứ tự lớp CHÍNH XÁC từ Colab của bạn
class_names = ['COVID-19', 'Phổi bình thường (Normal)', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

st.set_page_config(page_title="AI Phân loại X-quang Phổi (4 lớp)", layout="centered")

st.title("🫁 AI Nhận diện 4 lớp bệnh phổi từ X-quang")
st.markdown("---")

st.write("""
Phân loại:  
- COVID-19  
- Phổi bình thường (Normal)  
- Viêm phổi (Pneumonia)  
- Lao phổi (Tuberculosis)  
""")

st.error("⚠️ **Chỉ hỗ trợ tham khảo – Không thay thế bác sĩ!** Nếu ảnh bình thường, sẽ hiển thị 'Normal'.")

uploaded_file = st.file_uploader("Upload ảnh X-quang (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh upload", width=400)

    with st.spinner("Phân tích..."):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100

    st.markdown("---")

    predicted_name = class_names[predicted_idx]

    # Kết quả chính – Đặc biệt cho Normal
    if predicted_name == 'Phổi bình thường (Normal)':
        st.success(f"**Kết quả: {predicted_name}** (Không có dấu hiệu bệnh)")
    elif confidence >= 70:
        st.success(f"**Kết quả: {predicted_name}**")
    elif confidence >= 50:
        st.warning(f"**Kết quả: {predicted_name}** (Độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {predicted_name}** (Độ tin cậy thấp)")

    st.write(f"**Độ tin cậy: {confidence:.2f}%**")

    # Fix lỗi progress: Clamp giá trị + hiển thị an toàn
    st.write("### Xác suất chi tiết:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        progress_val = max(0.0, min(1.0, prob / 100))  # Clamp 0-1
        st.progress(progress_val)
        if i == predicted_idx:
            st.write(f"**{name}: {prob:.2f}%**")
        else:
            st.write(f"{name}: {prob:.2f}%")

    st.info("💡 **Khuyến nghị**: Kết quả chỉ mang tính tham khảo. Hãy đến bác sĩ để chẩn đoán chính xác!")
else:
    st.info("👆 Upload ảnh X-quang để kiểm tra.")
