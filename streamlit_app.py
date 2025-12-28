import streamlit as st
from huggingface_hub import hf_hub_download
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model từ Hugging Face (cache để nhanh)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Silver3424/lung-disease-4classes",  # THAY BẰNG REPO THẬT CỦA BẠN
        filename="lung_4_classes_focal.keras"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ['COVID-19', 'Phổi bình thường (Normal)', 'Viêm phổi (Pneumonia)', 'Lao phổi (Tuberculosis)']

st.title("🫁 AI Nhận diện 4 bệnh phổi từ X-quang")

uploaded_file = st.file_uploader("Upload ảnh X-quang", type=["jpg", "png", "jpeg"])

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

    predicted_name = class_names[predicted_idx]

    if predicted_name == 'Phổi bình thường (Normal)':
        st.success(f"**Kết quả: {predicted_name}**")
    else:
        st.warning(f"**Kết quả: {predicted_name}** (Độ tin cậy: {confidence:.2f}%)")

    st.write("### Xác suất chi tiết:")
    for i, name in enumerate(class_names):
        prob = predictions[i] * 100
        st.progress(prob / 100)
        st.write(f"{name}: {prob:.2f}%")

    st.info("💡 Kết quả chỉ tham khảo – Hãy đến bác sĩ!")
