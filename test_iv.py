import cv2
import numpy as np
import os
import dlib

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

# Load model Dlib
PREDICTOR_PATH = os.path.join(BASE_DIR, "CV_Lab", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("Không tìm thấy file shape_predictor_68_face_landmarks.dat!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ========================== HÀM XỬ LÝ ==========================

def get_landmarks(img, face, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    return np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)

def get_face_mask(img, landmarks, indices):
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    selected_points = landmarks[indices]
    points = cv2.convexHull(selected_points)
    cv2.fillConvexPoly(mask, points, 1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    return mask

def align_faces(src_landmarks, dst_landmarks):
    
    # Lấy mắt và mũi để tính toán tỉ lệ và góc xoay
    src_left_eye = np.mean(src_landmarks[36:42], axis=0)
    src_right_eye = np.mean(src_landmarks[42:48], axis=0)
    src_nose = src_landmarks[30]

    dst_left_eye = np.mean(dst_landmarks[36:42], axis=0)
    dst_right_eye = np.mean(dst_landmarks[42:48], axis=0)
    dst_nose = dst_landmarks[30]

    # Tính toán góc xoay khuôn mặt
    src_eye_angle = np.arctan2(src_right_eye[1] - src_left_eye[1], src_right_eye[0] - src_left_eye[0])
    dst_eye_angle = np.arctan2(dst_right_eye[1] - dst_left_eye[1], dst_right_eye[0] - dst_left_eye[0])
    rotation_angle = np.degrees(src_eye_angle - dst_eye_angle)

    # Tính toán tỉ lệ scale dựa trên khoảng cách giữa mắt và mũi
    src_distance = np.linalg.norm(src_left_eye - src_right_eye)
    dst_distance = np.linalg.norm(dst_left_eye - dst_right_eye)
    scale_factor = dst_distance / src_distance

    # Xây dựng ma trận biến đổi Affine
    M = cv2.getRotationMatrix2D((float(src_nose[0]), float(src_nose[1])), rotation_angle, scale_factor)
    M[0, 2] += float(dst_nose[0]) - float(src_nose[0])
    M[1, 2] += float(dst_nose[1]) - float(src_nose[1])
    return M

def multi_band_blending(src_img, dst_img, mask, num_levels=5):
    gp_mask = [mask.astype(np.float32)]
    print(gp_mask)
    for i in range(num_levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))
    print(gp_mask)

    gp_src = [src_img.astype(np.float32)]
    gp_dst = [dst_img.astype(np.float32)]
    for i in range(num_levels):
        gp_src.append(cv2.pyrDown(gp_src[-1]))
        gp_dst.append(cv2.pyrDown(gp_dst[-1]))

    lp_src = [gp_src[-1]]
    lp_dst = [gp_dst[-1]]
    for i in range(num_levels - 1, -1, -1):
        src_lap = cv2.subtract(gp_src[i], cv2.pyrUp(gp_src[i + 1], dstsize=gp_src[i].shape[:2][::-1]))
        dst_lap = cv2.subtract(gp_dst[i], cv2.pyrUp(gp_dst[i + 1], dstsize=gp_dst[i].shape[:2][::-1]))
        lp_src.append(src_lap)
        lp_dst.append(dst_lap)

    blended_pyramid = []
    for l_src, l_dst, g_mask in zip(lp_src, lp_dst, gp_mask[::-1]):
        blended = l_src * g_mask + l_dst * (1 - g_mask)
        blended_pyramid.append(blended)

    blended_img = blended_pyramid[0]
    for i in range(1, num_levels + 1):
        blended_img = cv2.pyrUp(blended_img, dstsize=blended_pyramid[i].shape[:2][::-1])
        blended_img = cv2.add(blended_img, blended_pyramid[i])

    return blended_img.astype(np.uint8)

# key_indices: Chọn các điểm khuôn mặt để xử lý
def swap_facial_features(src_img, dst_img, src_landmarks, dst_landmarks):
    key_indices = list(range(17, 27)) + list(range(27, 36)) + list(range(36, 48)) + list(range(48, 68))
    M = align_faces(dst_landmarks, src_landmarks)
    dst_img_aligned = cv2.warpAffine(dst_img, M, (src_img.shape[1], src_img.shape[0]))
    mask = get_face_mask(src_img, src_landmarks, key_indices)
    blended_face = multi_band_blending(dst_img_aligned, src_img, mask, num_levels=5)
    return blended_face

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

    left_eye = landmarks[36]  # Góc ngoài mắt trái
    right_eye = landmarks[45]  # Góc ngoài mắt phải
    nose = landmarks[30]  # Chóp mũi

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
    faces = detector(frame)
    #1 faces_img1 = detector(img1)

    #1 if len(faces) > 0 and len(faces_img1) > 0:
    if len(faces) > 0 and filter_image is not None:
        landmarks_frame = get_landmarks(frame, faces[0], predictor)
        #1 landmarks_img1 = get_landmarks(img1, faces_img1[0], predictor) 
        #1 swapped_face = swap_facial_features(frame, img1, landmarks_frame, landmarks_img1)
        #1 cv2.imshow("Face Swap Real-Time", swapped_face)
        applied_filter = apply_face_filter(frame, filter_image, landmarks_frame, scale_factor=2, position_factor=1.2)
        cv2.imshow("Face Swap Real-Time", applied_filter)
    else:
        cv2.imshow("Face Swap Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
