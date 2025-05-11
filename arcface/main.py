import os
import cv2
import glob
import numpy as np
import pandas as pd

# GAN training
import torch
import torch.nn as nn
import torch.optim as optim

from model_processing.generate_dataset import generate_dataset
from model_processing.data_preparation import data_preparation
from model_processing.gan import FaceSwapGenerator, FaceSwapDiscriminator

# DIR
IMG_DIR = "CelebA/img_align_celeba"
SAVED_DIR = "CelebA/processed_faces"

# ========== Load CelebA dataset =============
celeba_bbox_path = "CelebA/list_bbox_celeba.csv"
# celeba_landmarks_path = "CelebA/face_landmarks.csv"

# Đọc danh sách bounding box
bbox_df = pd.read_csv(celeba_bbox_path, dtype={"image_id": str})

# # Đọc danh sách landmark 68 điểm
# landmarks_df = pd.read_csv(celeba_landmarks_path, dtype={"image_id": str})

bbox_df.set_index("image_id", inplace=True)
# landmarks_df.set_index("image_id", inplace=True)
bbox_df.index = bbox_df.index.str.strip()  # Xóa khoảng trắng
# landmarks_df.index = bbox_df.index.str.strip()  # Xóa khoảng trắng

print(bbox_df.head())
# print(landmarks_df.head())

# ========== Crop faces from CelebA =============
generate_dataset(IMG_DIR, SAVED_DIR, bbox_df)

# ========== Data Preparation =============
dataset = data_preparation(SAVED_DIR)

# ========== Training GAN Model =============
# Khởi tạo model
G = FaceSwapGenerator()
D = FaceSwapDiscriminator()

# Loss function và optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 10
for epoch in range(num_epochs):
    for img_path in dataset:  # Chạy thử 100 ảnh
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # Chuẩn hóa ảnh

        # **✅ Lấy kích thước đầu ra thực tế của Discriminator**
        real_output = D(img)
        fake_img = G(img)
        fake_output = D(fake_img.detach())

        # **✅ Sửa nhãn để có cùng kích thước với đầu ra của D**
        real_labels = torch.ones_like(real_output)  # Nhãn thật có cùng kích thước với real_output
        fake_labels = torch.zeros_like(fake_output)  # Nhãn giả có cùng kích thước với fake_output

        # **Train Discriminator**
        optimizer_D.zero_grad()
        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # **Train Generator**
        optimizer_G.zero_grad()
        fake_output = D(fake_img)  # Discriminator đánh giá ảnh giả
        g_loss = criterion(fake_output, real_labels)  # Generator muốn lừa Discriminator nên nhãn vẫn là 1
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

print("✅ Training hoàn tất!")

torch.save(G, "face_swap_generator_full.pth")
torch.save(D, "face_swap_discriminator_full.pth")
print("Training và Lưu hoàn tất!")

# ========== Testing =============
# Bước 1: Tải modelmodel
# G = torch.load("face_swap_generator_full.pth", weights_only=False)
# D = torch.load("face_swap_discriminator_full.pth", weights_only=False)
# G.eval()
# D.eval()

# def swap_face(image_path, model):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))
#     img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
#     swapped_face = model(img_tensor).detach().squeeze().permute(1, 2, 0).numpy() * 255
#     swapped_face = swapped_face.astype(np.uint8)

#     cv2.imshow("Swapped Face", swapped_face)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Test ảnh bất kỳ
# swap_face("CelebA/processed_faces/000007.jpg", G)