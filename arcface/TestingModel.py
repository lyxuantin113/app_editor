import os
import cv2
import glob
import numpy as np
import pandas as pd
import dlib

import torch
import torch.nn as nn
import torch.optim as optim

# Bước 1: Tải modelmodel
G = torch.load("face_swap_generator_full.pth", weights_only=False)
D = torch.load("face_swap_discriminator_full.pth", weights_only=False)
G.eval()
D.eval()

# Bước 2: Định danh khuôn mặt và trích xuất đặc trưng
detector = dlib.get_frontal_face_detector()

def detect_face(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)

    if len(faces) == 0:
        print(f"Không tìm thấy khuôn mặt trong {image_path}")
        return None, None

    # Lấy khuôn mặt đầu tiên
    x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
    face_crop = img[y:y+h, x:x+w]

    print("✅ Hoàn tất :", "detect_face")
    return face_crop, (x, y, w, h)

# Lấy khuôn mặt từ ảnh nguồn và ảnh đích
source_face, source_bbox = detect_face("../CV_Lab/f1.jpg")
target_face, target_bbox = detect_face("../CV_Lab/f2.jpg")

# Bước 3: Tiền xử lý khuôn mặt trước khi đưa vào mô hình
def preprocess_face(face):
    face = cv2.resize(face, (128, 128))
    face = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    print("✅ Hoàn tất :", "preprocess_face")
    return face

source_face_tensor = preprocess_face(source_face)
target_face_tensor = preprocess_face(target_face)

# Bước 4: Swap khuôn mặt bằng GAN
with torch.no_grad():  # Không cần tính toán gradient
    swapped_face_tensor = G(source_face_tensor)

# Chuyển đổi tensor thành ảnh
swapped_face = swapped_face_tensor.detach().squeeze().permute(1, 2, 0).numpy() * 255
swapped_face = swapped_face.astype(np.uint8)
print("✅ Hoàn tất :", "Bước 4")

# Bước 5: Ghép khuôn mặt mới vào ảnh đích
def blend_faces(original_image, swapped_face, bbox):
    x, y, w, h = bbox
    swapped_face_resized = cv2.resize(swapped_face, (w, h))
    
    # Tạo mặt nạ để ghép ảnh
    mask = np.zeros_like(original_image)
    mask[y:y+h, x:x+w] = 1
    
    # Ghép ảnh bằng phương pháp alpha blending
    blended_image = original_image.copy()
    blended_image[y:y+h, x:x+w] = swapped_face_resized * mask[y:y+h, x:x+w] + original_image[y:y+h, x:x+w] * (1 - mask[y:y+h, x:x+w])

    print("✅ Hoàn tất :", "Blend")
    return blended_image

# Đọc lại ảnh đích để chèn khuôn mặt mới
target_image = cv2.imread("../CV_Lab/f2.jpg")
result = blend_faces(target_image, swapped_face, target_bbox)

# Hiển thị ảnh kết quả
cv2.imshow("Swapped Face", result)
cv2.waitKey(0)
cv2.destroyAllWindows()