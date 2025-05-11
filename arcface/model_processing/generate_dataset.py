import os
import cv2
import dlib
import torch

def generate_dataset(IMG_DIR, SAVE_DIR, bbox_df):
    """
    Cắt ảnh khuôn mặt từ bbox và căn chỉnh chính xác hơn dựa trên landmark.
    """
    detector = dlib.get_frontal_face_detector()

    for img_name in bbox_df.index:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = detector(img, 1)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        face_crop = img[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (128, 128))

        save_path = os.path.join(SAVE_DIR, img_name)
        cv2.imwrite(save_path, face_crop)

    print("✅ Dữ liệu khuôn mặt đã được chuẩn bị.")
