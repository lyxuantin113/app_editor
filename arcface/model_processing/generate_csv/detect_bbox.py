import os
import cv2
import dlib
import pandas as pd

# Đường dẫn đến thư mục chứa ảnh CelebA
IMG_DIR = "CelebA/img_align_celeba/"
SAVE_DIR = "CelebA/a/"
CSV_PATH = "CelebA/face_bboxes.csv"

# Kiểm tra và tạo thư mục lưu ảnh đã xử lý nếu chưa có
os.makedirs(SAVE_DIR, exist_ok=True)

# Khởi tạo bộ phát hiện khuôn mặt của Dlib
detector = dlib.get_frontal_face_detector()

# Danh sách lưu kết quả
data = []

# Duyệt qua tất cả ảnh trong thư mục
for img_name in sorted(os.listdir(IMG_DIR))[:100]:  # Chỉ lấy 100 ảnh đầu tiên để kiểm tra
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

    # Lấy bbox khuôn mặt đầu tiên
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Cắt khuôn mặt
    face_crop = img[y:y+h, x:x+w]

    # Resize về 128x128
    face_crop = cv2.resize(face_crop, (128, 128))

    # Lưu ảnh đã xử lý
    save_path = os.path.join(SAVE_DIR, img_name)
    cv2.imwrite(save_path, face_crop)

    # Thêm thông tin vào danh sách
    data.append([img_name, x, y, w, h])

# Lưu dữ liệu vào file CSV
df = pd.DataFrame(data, columns=["image_id", "x", "y", "width", "height"])
df.to_csv(CSV_PATH, index=False)

print(f"Hoàn tất xử lý, dữ liệu được lưu tại: {CSV_PATH}")
