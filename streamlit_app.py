import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import zipfile
import os

# =============================================
# GIẢI NÉN MÔ HÌNH TỪ FILE ZIP (nếu chưa có) – CHO MODEL MỚI 4 LỚP
# =============================================
model_file = "lung_4_classes_model.keras"
zip_file = "lung_4_classes_model.zip"  # Tên file zip mới nếu bạn upload zip (nếu không zip thì bỏ phần này)

if not os.path.exists(model_file):
    if os.path.exists(zip_file):
        st.write("🔄 Đang giải nén mô hình mới (4 lớp) từ file zip... (chỉ lần đầu, mất khoảng 20-60 giây)")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Giải nén mô hình thành công!")
    else:
        st.error(f"❌ Không tìm thấy file mô hình hoặc zip: {model_file} / {zip_file}")
        st.stop()  # Dừng app nếu không có model
else:
    st.write("✅ Mô hình 4 lớp đã sẵn sàng (đã được giải nén từ trước).")

# =============================================
# LOAD MÔ HÌNH MỚI (4 LỚP – KHÔNG DÙNG CACHE ĐỂ TRÁNH LỖI)
# =============================================
with st.spinner("Đang tải mô hình AI mới (4 lớp)... (lần đầu có thể mất 20-40 giây)"):
    model = tf.keras.models.load_model(model_file)

# =============================================
# THỨ TỰ LỚP – BẮT BUỘC ĐÚNG VỚI CLASS_INDICES IN RA Ở CELL 3
# =============================================
# Sửa chính xác theo output của Colab (thường alphabet: COVID19=0, NORMAL=1, PNEUMONIA=2, TURBERCULOSIS=3)
class_names = ['COVID-19', 'Phổi bình thường (Normal)', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

# =============================================
# GIAO DIỆN STREAMLIT – ĐÃ CẬP NHẬT CHO 4 LỚP
# =============================================
st.set_page_config(page_title="Nhận diện bệnh phổi từ X-quang (4 lớp)", layout="centered")

st.title("🫁 Nhận diện 4 lớp bệnh phổi từ ảnh X-quang ngực")
st.markdown("---")

st.write("""
Ứng dụng sử dụng mô hình Deep Learning (MobileNetV2) để phân loại ảnh X-quang thành một trong 4 lớp:
- **COVID-19**
- **Phổi bình thường (Normal)**
- **Viêm phổi (Pneumonia)**
- **Lao phổi (Tuberculosis)**
""")

st.error("⚠️ Đây chỉ là công cụ hỗ trợ AI – KHÔNG thay thế chẩn đoán của bác sĩ! Ảnh bình thường sẽ được nhận là 'Normal'.")

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

    # Kết quả chính – Đặc biệt xử lý cho Normal
    predicted_name = class_names[predicted_idx]
    if predicted_name == 'Phổi bình thường (Normal)':
        st.success(f"**Kết quả dự đoán: {predicted_name}**")
    elif confidence >= 70:
        st.success(f"**Kết quả dự đoán: {predicted_name}**")
    elif confidence >= 50:
        st.warning(f"**Kết quả dự đoán: {predicted_name}** (độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {predicted_name}** (độ tin cậy thấp)")

    st.write(f"**Độ tin cậy: {confidence:.2f}%**")

    # Chi tiết xác suất – ĐÃ FIX LỖI PROGRESS (clamp giá trị 0-1)
    st.write("### Xác suất chi tiết:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        progress_value = max(0.0, min(1.0, prob / 100))  # Clamp để tránh lỗi StreamlitAPIException
        st.progress(progress_value)
        if i == predicted_idx:
            st.write(f"**{name}: {prob:.2f}%**")
        else:
            st.write(f"{name}: {prob:.2f}%")

    st.info("💡 Khuyến nghị: Hãy mang kết quả này đến bác sĩ chuyên khoa để được chẩn đoán chính xác!")
else:
    st.info("👆 Hãy upload một ảnh X-quang ngực để bắt đầu phân tích.")
