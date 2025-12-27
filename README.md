# Lung 4 Classes Classification Model

Model Keras đã train sẵn để phân loại hình ảnh phổi thành 4 lớp.

## Thông tin model
- File: `lung_4_classes_fixed.keras`
- Kích thước: ~53 MB
- Framework: TensorFlow / Keras

## Cách sử dụng (tự động tải model từ Google Drive)

Bạn chỉ cần chạy đoạn code sau, model sẽ tự động được tải về máy lần đầu tiên:

```python
import tensorflow as tf
import gdown  # Thư viện giúp tải file từ Google Drive dễ dàng

# Link direct download từ Google Drive
url = "https://drive.google.com/uc?export=download&id=1LpZeK3Em1hDxNd4rXzhgm9huvdGuakAr"

# Tải model (chỉ tải 1 lần, lần sau sẽ dùng file đã có)
gdown.download(url, "lung_4_classes_fixed.keras", quiet=False)

# Load model vào chương trình
model = tf.keras.models.load_model("lung_4_classes_fixed.keras")

# Kiểm tra model
model.summary()
