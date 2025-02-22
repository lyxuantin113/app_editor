import cv2
import numpy as np
import os
import dlib

from insightface.app import FaceAnalysis

# ========================== SETUP ==========================

# Cấu hình đường dẫn cho ảnh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "CV_Lab")

# Đọc ảnh đích (f1.jpg - ảnh nền để chèn khuôn mặt từ webcam)
img1_path = os.path.join(IMG_DIR, "f3.jpg")
if not os.path.exists(img1_path):
    raise FileNotFoundError("Không tìm thấy ảnh f1.jpg trong thư mục CV_Lab!")

img1 = cv2.imread(img1_path)
img1 = cv2.resize(img1, (256, 256))

# SCRFDSCRFD
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ========================== HÀM XỬ LÝ ==========================

def get_landmarks(img, face):
    """
    Lấy landmarks từ SCRFD
    """
    return np.array(face['kps'], dtype=np.int32)

def get_face_mask(img, landmarks, indices):
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    selected_points = landmarks[indices]
    points = cv2.convexHull(selected_points)
    cv2.fillConvexPoly(mask, points, 1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    return mask

# ========================== Filtering ==========================

def apply_face_filter(image, filter_img, landmarks, scale_factor=2.2, position_factor=0.5):
    """
    Áp dụng filter (kính/mũ) lên khuôn mặt.

    Parameters:
    - image: Ảnh gốc.
    - filter_img: Ảnh filter (có thể là kính hoặc mũ).
    - landmarks: Danh sách tọa độ landmarks của khuôn mặt.
    - scale_factor: Độ rộng của filter so với khoảng cách hai mắt.
        . kính: 2.1
        . nón : 2.2
    - position_factor: Hệ số điều chỉnh vị trí filter theo chiều dọc.
        . kính: 0.65
        . nón : 1.2
    """

    left_eye = landmarks[0]  # Góc ngoài mắt trái
    right_eye = landmarks[1]  # Góc ngoài mắt phải
    nose = landmarks[2]  # Chóp mũi

    # 1️⃣ Tính khoảng cách giữa hai mắt
    eye_width = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    # 2️⃣ Xác định kích thước filter
    filter_width = int(eye_width * scale_factor)
    aspect_ratio = filter_img.shape[0] / filter_img.shape[1]
    filter_height = int(filter_width * aspect_ratio)

    # 3️⃣ Resize filter
    filter_resized = cv2.resize(filter_img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

    # 4️⃣ Tính góc xoay của filter
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = -np.degrees(np.arctan2(dy, dx))

    center = (filter_width // 2, filter_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    filter_rotated = cv2.warpAffine(filter_resized, rotation_matrix, (filter_width, filter_height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 5️⃣ Xác định vị trí đặt filter
    anchor_x = int((left_eye[0] + right_eye[0]) / 2) - filter_width // 2
    anchor_y = int(nose[1] - filter_height * position_factor)

    # 6️⃣ Xác định vùng hợp lệ trong ảnh gốc
    x1, y1 = anchor_x, anchor_y
    x2, y2 = anchor_x + filter_width, anchor_y + filter_height

    # Cắt filter nếu vượt khỏi ảnh
    filter_x1 = max(0, -x1)
    filter_x2 = filter_width - max(0, x2 - image.shape[1])
    filter_y1 = max(0, -y1)
    filter_y2 = filter_height - max(0, y2 - image.shape[0])

    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    # Cắt filter để khớp với kích thước hợp lệ
    filter_cropped = filter_rotated[filter_y1:filter_y2, filter_x1:filter_x2]

    # Xử lý kênh alpha
    if filter_cropped.shape[-1] == 4:
        mask = filter_cropped[:, :, 3]
    else:
        mask = np.ones(filter_cropped.shape[:2], dtype=np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)

    # Vùng ảnh đích
    roi = image[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(filter_cropped[:, :, :3], filter_cropped[:, :, :3], mask=mask)
    result = cv2.add(roi_bg, roi_fg)

    # Chèn vào ảnh gốc
    image[y1:y2, x1:x2] = result

    return image


# ========================== REAL-TIME FACE SWAP ==========================
filter_image = cv2.imread(os.path.join(IMG_DIR, "hat3.png"), cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces = app.get(frame)
    if len(faces) > 0:
        landmarks_frame = get_landmarks(frame, faces[0])
        applied_filter = apply_face_filter(frame, filter_image, landmarks_frame, scale_factor=2.5, position_factor=1.2)
        cv2.imshow("Face Swap Real-Time", applied_filter)
    else:
        cv2.imshow("Face Swap Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
