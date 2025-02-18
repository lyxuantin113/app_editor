import os
import cv2
import dlib
import pandas as pd

# ==================== SETUP ====================
# Đường dẫn đến thư mục chứa ảnh CelebA
IMG_DIR = "CelebA/img_align_celeba/"
CSV_PATH = "CelebA/face_landmarks.csv"

# Load bộ phát hiện khuôn mặt và nhận diện landmark của Dlib
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../CV_Lab/shape_predictor_68_face_landmarks.dat"  # Đường dẫn file model Dlib
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ==================== TẠO DATASET LANDMARK ====================
data = []

# Duyệt qua 100 ảnh đầu tiên
for img_name in sorted(os.listdir(IMG_DIR))[:100]:
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Lỗi khi đọc ảnh: {img_name}")
        continue

    # Phát hiện khuôn mặt
    faces = detector(img, 1)
    if len(faces) == 0:
        print(f"Không tìm thấy khuôn mặt trong ảnh: {img_name}")
        continue

    # Lấy khuôn mặt đầu tiên
    face = faces[0]

    # Dự đoán landmark
    landmarks = predictor(img, face)

    # Lấy 5 điểm quan trọng
    lefteye_x, lefteye_y = landmarks.part(36).x, landmarks.part(36).y  # Mắt trái
    righteye_x, righteye_y = landmarks.part(45).x, landmarks.part(45).y  # Mắt phải
    nose_x, nose_y = landmarks.part(30).x, landmarks.part(30).y  # Mũi
    leftmouth_x, leftmouth_y = landmarks.part(48).x, landmarks.part(48).y  # Góc trái miệng
    rightmouth_x, rightmouth_y = landmarks.part(54).x, landmarks.part(54).y  # Góc phải miệng

    # Thêm dữ liệu vào danh sách
    data.append([img_name, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y])

# ==================== LƯU DỮ LIỆU LANDMARK VÀO CSV ====================
columns = ["image_id", "lefteye_x", "lefteye_y", "righteye_x", "righteye_y", "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y"]
df = pd.DataFrame(data, columns=columns)

# Lưu file CSV
df.to_csv(CSV_PATH, index=False)

print(f"✅ Hoàn tất xử lý, dữ liệu landmark được lưu tại: {CSV_PATH}")
