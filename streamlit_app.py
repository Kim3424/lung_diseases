import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import zipfile
import os

# =============================================
# GIẢI NÉN MÔ HÌNH TỪ FILE ZIP (nếu chưa có)
# =============================================
model_file = "lung_3_diseases_model.keras"
zip_file = "lung_3_diseases_model.zip"  # Tên file zip bạn đã upload lên GitHub

if not os.path.exists(model_file):
    if os.path.exists(zip_file):
        st.write("🔄 Đang giải nén mô hình từ file zip... (chỉ lần đầu, mất khoảng 20-60 giây)")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Giải nén mô hình thành công!")
    else:
        st.error(f"❌ Không tìm thấy file zip mô hình: {zip_file}")
        st.stop()  # Dừng app nếu không có model
else:
    st.write("✅ Mô hình đã sẵn sàng (đã được giải nén từ trước).")

# =============================================
# LOAD MÔ HÌNH (không dùng cache để tránh lỗi hash)
# =============================================
with st.spinner("Đang tải mô hình AI... (lần đầu có thể mất 20-40 giây)"):
    model = tf.keras.models.load_model(model_file)

# =============================================
# THỨ TỰ LỚP – BẮT BUỘC ĐÚNG VỚI COLAB
# =============================================
# Nếu class_indices ở Colab in ra khác thứ tự này thì bạn sửa lại cho đúng nhé!
class_names = ['COVID-19', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

# =============================================
# GIAO DIỆN STREAMLIT
# =============================================
st.set_page_config(page_title="Nhận diện bệnh phổi từ X-quang", layout="centered")

st.title("🫁 Nhận diện 3 bệnh phổi từ ảnh X-quang ngực")
st.markdown("---")

st.write("""
Ứng dụng sử dụng mô hình Deep Learning (MobileNetV2) để phân loại ảnh X-quang thành một trong 3 bệnh:
- **COVID-19**
- **Viêm phổi (Pneumonia)**
- **Lao phổi (Tuberculosis)**
""")

st.error("⚠️ Đây chỉ là công cụ hỗ trợ AI – KHÔNG thay thế chẩn đoán của bác sĩ!")

# Upload ảnh
uploaded_file = st.file_uploader("Upload ảnh X-quang (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã upload", width=400)

    # Dự đoán
    with st.spinner("Đang phân tích ảnh bằng AI..."):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100

    st.markdown("---")

    # Kết quả chính
    if confidence >= 70:
        st.success(f"**Kết quả dự đoán: {class_names[predicted_idx]}**")
    elif confidence >= 50:
        st.warning(f"**Kết quả dự đoán: {class_names[predicted_idx]}** (độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {class_names[predicted_idx]}** (độ tin cậy thấp)")

    st.write(f"**Độ tin cậy: {confidence:.2f}%**")

    # Chi tiết xác suất
    st.write("### Xác suất chi tiết:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        st.progress(prob / 100)
        if i == predicted_idx:
            st.write(f"**{name}: {prob:.2f}%**")
        else:
            st.write(f"{name}: {prob:.2f}%")

    st.info("💡 Khuyến nghị: Hãy mang kết quả này đến bác sĩ chuyên khoa để được chẩn đoán chính xác!")
else:
    st.info("👆 Hãy upload một ảnh X-quang ngực để bắt đầu phân tích.")
