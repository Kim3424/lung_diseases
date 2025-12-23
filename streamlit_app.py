import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load mô hình (định dạng mới .keras)
@st.cache_resource  # Chỉ load 1 lần để nhanh hơn
def load_model():
    return tf.keras.models.load_model('lung_3_diseases_model.keras')

model = load_model()

# Thứ tự lớp – BẮT BUỘC PHẢI ĐÚNG với class_indices in ra ở Cell 3 Colab!
# Ví dụ phổ biến từ dataset này: COVID19 = 0, PNEUMONIA = 1, TURBERCULOSIS = 2
class_names = ['COVID-19', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

# Giao diện Streamlit
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
uploaded_file = st.file_uploader("Upload ảnh X-quang (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã upload", width=400)

    # Preprocess
    with st.spinner("Đang phân tích ảnh..."):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100

    # Kết quả chính
    st.markdown("---")
    if confidence > 70:
        st.success(f"**Kết quả dự đoán: {class_names[predicted_idx]}**")
    elif confidence > 50:
        st.warning(f"**Kết quả dự đoán: {class_names[predicted_idx]}** (độ tin cậy trung bình)")
    else:
        st.error(f"**Kết quả không rõ ràng: {class_names[predicted_idx]}** (độ tin cậy thấp)")

    st.write(f"**Độ tin cậy: {confidence:.2f}%**")

    # Chi tiết xác suất từng lớp
    st.write("### Xác suất chi tiết:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        if i == predicted_idx:
            st.progress(prob / 100)
            st.write(f"**{name}: {prob:.2f}%**")
        else:
            st.progress(prob / 100)
            st.write(f"{name}: {prob:.2f}%")

    st.info("💡 Khuyến nghị: Hãy mang kết quả này đến bác sĩ chuyên khoa để được chẩn đoán chính xác!")
