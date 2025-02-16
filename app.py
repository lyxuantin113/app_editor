import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from insightface.app import FaceAnalysis
import dlib

# ========================== SETUP ==========================

# Cấu hình đường dẫn cho ảnh và mô hình Dlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "CV_Lab")

# Đọc ảnh nguồn và ảnh đích
img1_path = os.path.join(IMG_DIR, "f1.jpg")
img2_path = os.path.join(IMG_DIR, "f2.jpg")

# Kiểm tra xem ảnh có tồn tại không
if not os.path.exists(img1_path) or not os.path.exists(img2_path):
    raise FileNotFoundError("Không tìm thấy ảnh f1.jpg hoặc f2.jpg trong thư mục CV_Lab!")

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Resize ảnh để đồng bộ kích thước
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# Làm mờ ảnh để giảm noise trước khi xử lý
img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)

# Tăng cường độ sắc nét
alpha = 1.5
sharpened1 = cv2.addWeighted(img1, 1 + alpha, img1_blur, -alpha, 0)
sharpened2 = cv2.addWeighted(img2, 1 + alpha, img2_blur, -alpha, 0)

# ========================== LOAD MODEL ==========================

# Kiểm tra model Dlib có tồn tại không
PREDICTOR_PATH = os.path.join(BASE_DIR, "CV_Lab", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("Không tìm thấy file shape_predictor_68_face_landmarks.dat!")

# Load mô hình phát hiện khuôn mặt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ========================== LANDMARK DETECTION ==========================

# 1. Resize

def resize_image_to_match(source_img, target_img):
    target_h, target_w = target_img.shape[:2]
    return cv2.resize(source_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# 2. Hàm lấy Landmark lấy 68 điểm mốc -> shape (N,2), N = 68

def get_landmarks(img, face, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    return np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)

# 3. Tạo mặt nạ từ Landmark

def get_face_mask(img, landmarks, indices):
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    selected_points = landmarks[indices]
    points = cv2.convexHull(selected_points)
    cv2.fillConvexPoly(mask, points, 1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    return mask

# 4. Tự động căn chỉnh khuôn mặt

def align_faces(src_landmarks, dst_landmarks):
    # Chỉ lấy các điểm mắt, mũi và miệng để căn chỉnh
    key_indices = list(range(17, 27)) + list(range(27, 36)) + list(range(48, 68))

    src_key_points = src_landmarks[key_indices]
    dst_key_points = dst_landmarks[key_indices]

    # Tính trung tâm các đặc điểm cần hoán đổi
    src_center = np.mean(src_key_points, axis=0)
    dst_center = np.mean(dst_key_points, axis=0)

    # Tính tỷ lệ giữa khoảng cách mắt và miệng, chân mày để điều chỉnh kích thước
    src_eye_distance = np.linalg.norm(src_landmarks[36] - src_landmarks[47])
    dst_eye_distance = np.linalg.norm(dst_landmarks[36] - dst_landmarks[47])

    src_mouth_distance = np.linalg.norm(src_landmarks[48] - src_landmarks[54])
    dst_mouth_distance = np.linalg.norm(dst_landmarks[48] - dst_landmarks[54])

    src_brow_distance = np.linalg.norm(src_landmarks[17] - src_landmarks[26])
    dst_brow_distance = np.linalg.norm(dst_landmarks[17] - dst_landmarks[26])

    scale_brow = dst_brow_distance / src_brow_distance
    scale_eye = dst_eye_distance / src_eye_distance
    scale_mouth = dst_mouth_distance / src_mouth_distance
    scale_factor = (scale_eye + scale_mouth + scale_brow) / 3

    # Tạo ma trận biến đổi với tỷ lệ và dịch chuyển để đặt vào trung tâm
    M = cv2.getRotationMatrix2D(tuple(src_center), 0, scale_factor)
    M[0, 2] += dst_center[0] - src_center[0]
    M[1, 2] += dst_center[1] - src_center[1]

    return M

# 5. Hoán đổi mắt, mũi , miệng và mày

def swap_facial_features(src_img, dst_img, src_landmarks, dst_landmarks):
    # Chỉ lấy các điểm mắt, mũi và miệng
    key_indices = list(range(17, 27)) + list(range(27, 36)) + list(range(36, 48)) + list(range(48, 68))

    # Căn chỉnh khuôn mặt
    M = align_faces(src_landmarks, dst_landmarks)
    src_img_aligned = cv2.warpAffine(src_img, M, (dst_img.shape[1], dst_img.shape[0]))

    # Tạo mặt nạ từ các điểm đặc trưng
    mask = get_face_mask(dst_img, dst_landmarks, key_indices)

    # ###
    # Tạo mask riêng cho từng vùng thay vì toàn bộ như dòng trên
    # brows_indices = list(range(17, 27))
    # eyes_right_indices = list(range(42, 48))
    # eyes_left_indices = list(range(36, 42))
    # nose_indices = list(range(27, 36))
    # mouth_indices = list(range(48, 68))

    # # Tạo mặt nạ tổng hợp từ các vùng
    # mask_brows = get_face_mask(dst_img, dst_landmarks, brows_indices)
    # mask_eyes_right = get_face_mask(dst_img, dst_landmarks, eyes_right_indices)
    # mask_eyes_left = get_face_mask(dst_img, dst_landmarks, eyes_left_indices)
    # mask_nose = get_face_mask(dst_img, dst_landmarks, nose_indices)
    # mask_mouth = get_face_mask(dst_img, dst_landmarks, mouth_indices)

    # # Kết hợp các mask
    # mask = mask_brows + mask_eyes_right + mask_eyes_left + mask_nose + mask_mouth
    # ###

    # Tạo ảnh chứa các đặc điểm đã được hoán đổi
    swapped_face = np.copy(dst_img)
    swapped_face = np.where(mask == 1, src_img_aligned, swapped_face)

    # Làm mờ viền và blend khuôn mặt bằng Multi-band Blending
    blended_face = multi_band_blending(swapped_face, dst_img, mask, num_levels=5)

    # # Xử lý hậu kỳ: Phát hiện và loại bỏ các vùng biên không mong muốn
    # cleaned_face = remove_unwanted_edges(blended_face, mask)

    # Làm mờ viền một lần nữa để đảm bảo hòa quyện tự nhiên
    # final_output = selective_blur(blended_face, mask, blur_radius=15)

    return blended_face

# 6. Blend tự nhiên

def seamless_clone(src_face, dst_img, mask, center):
    return cv2.seamlessClone(src_face.astype(np.uint8), dst_img, (mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)

# 8. Multi-band Blending

def multi_band_blending(src_img, dst_img, mask, num_levels=5):

    gp_mask = [mask.astype(np.float32)]
    for i in range(num_levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

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


# ========================== MAIN PROCESS ==========================

# 9. Thực hiện Face Swap

def face_swap(src_img, dst_img):
    src_img_resized = resize_image_to_match(src_img, dst_img)

    faces_src = detector(cv2.cvtColor(src_img_resized, cv2.COLOR_BGR2GRAY))
    faces_dst = detector(cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY))

    if len(faces_src) == 0 or len(faces_dst) == 0:
        print("Không tìm thấy khuôn mặt trong một trong hai ảnh!")
        exit()

    landmarks_src = get_landmarks(src_img_resized, faces_src[0], predictor)
    landmarks_dst = get_landmarks(dst_img, faces_dst[0], predictor)

    # Hoán đổi mắt, mũi và miệng
    output = swap_facial_features(src_img_resized, dst_img, landmarks_src, landmarks_dst)

    return output

# ========================== EXECUTION ==========================

print("🔄 Đang hoán đổi khuôn mặt...")
result = face_swap(sharpened1, sharpened2)

# Hiển thị kết quả
plt.figure(figsize=(8, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(sharpened1, cv2.COLOR_BGR2RGB))
plt.title("Ảnh nguồn")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(sharpened2, cv2.COLOR_BGR2RGB))
plt.title("Ảnh đích")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Kết quả Face Swap")

plt.show()