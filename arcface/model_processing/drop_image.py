import os
import cv2

def drop_face(IMG_DIR, SAVE_DIR, bbox_df, landmark_df):
    """
    Cắt ảnh khuôn mặt từ bbox và căn chỉnh chính xác hơn dựa trên landmark.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)  # Tạo thư mục lưu nếu chưa có

    for img_name in bbox_df.index:  # Chỉ lấy 100 ảnh để kiểm tra
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Không thể đọc ảnh: {img_path}")
            continue  # Nếu ảnh lỗi, bỏ qua

        try:
            # Lấy tọa độ bbox từ bbox_df
            x, y, w, h = bbox_df.loc[img_name].astype(int)

            # Lấy tọa độ landmark từ landmark_df
            landmarks = landmark_df.loc[img_name, ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y", 
                                                   "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", 
                                                   "rightmouth_x", "rightmouth_y"]].astype(int)

            # Tính toán trung tâm khuôn mặt dựa vào mắt, mũi, miệng
            center_x = (landmarks["lefteye_x"] + landmarks["righteye_x"] + landmarks["nose_x"] + 
                        landmarks["leftmouth_x"] + landmarks["rightmouth_x"]) // 5
            center_y = (landmarks["lefteye_y"] + landmarks["righteye_y"] + landmarks["nose_y"] + 
                        landmarks["leftmouth_y"] + landmarks["rightmouth_y"]) // 5
            
            # Mở rộng bbox để chứa đầy đủ khuôn mặt
            new_size = int(max(w, h) * 1.2)  # Tăng 20% so với bbox gốc
            x1, y1 = max(center_x - new_size // 2, 0), max(center_y - new_size // 2, 0)
            x2, y2 = min(center_x + new_size // 2, img.shape[1]), min(center_y + new_size // 2, img.shape[0])

            # Cắt ảnh khuôn mặt
            face_crop = img[y1:y2, x1:x2]

            if face_crop.size == 0:
                print(f"⚠️ Bounding box quá nhỏ hoặc lỗi: {img_name} ({x1}, {y1}, {x2-x1}, {y2-y1})")
                continue  # Bỏ qua ảnh lỗi

            # Resize ảnh về kích thước chuẩn (128x128)
            face_crop = cv2.resize(face_crop, (128, 128))

            # Lưu ảnh đã xử lý
            save_path = os.path.join(SAVE_DIR, img_name)
            cv2.imwrite(save_path, face_crop)

        except KeyError as e:
            print(f"⚠️ Thiếu dữ liệu cho ảnh {img_name}: {e}")
            continue

    print("✅ Hoàn tất cắt khuôn mặt và lưu vào:", SAVE_DIR)
