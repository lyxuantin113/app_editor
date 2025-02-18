import os
import cv2
import glob
import numpy as np
import pandas as pd

# GAN training
import torch
import torch.nn as nn
import torch.optim as optim

from model_processing.drop_image import drop_face
from model_processing.data_preparation import data_preparation
from model_processing.gan import FaceSwapGenerator, FaceSwapDiscriminator

# # DIR
# IMG_DIR = "CelebA/img_align_celeba"
# SAVED_DIR = "CelebA/processed_faces"

# # ========== Load CelebA dataset =============
# celeba_bbox_path = "CelebA/face_bboxes.csv"
# celeba_landmarks_path = "CelebA/face_landmarks.csv"

# # Đọc danh sách bounding box
# bbox_df = pd.read_csv(celeba_bbox_path, dtype={"image_id": str})

# # Đọc danh sách landmark 68 điểm
# landmarks_df = pd.read_csv(celeba_landmarks_path, dtype={"image_id": str})

# bbox_df.set_index("image_id", inplace=True)
# landmarks_df.set_index("image_id", inplace=True)
# bbox_df.index = bbox_df.index.str.strip()  # Xóa khoảng trắng
# landmarks_df.index = bbox_df.index.str.strip()  # Xóa khoảng trắng

# # print(bbox_df.head())
# # print(landmarks_df.head())

# # ========== Crop faces from CelebA =============
# drop_face(IMG_DIR, SAVED_DIR, bbox_df, landmarks_df)

# # ========== Data Preparation =============
# dataset = data_preparation(SAVED_DIR)

# # ========== Training GAN Model =============
# # Khởi tạo model
# G = FaceSwapGenerator()
# D = FaceSwapDiscriminator()

# # Loss function và optimizer
# criterion = nn.BCELoss()
# optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# num_epochs = 10
# for epoch in range(num_epochs):
#     for img_path in dataset:  # Chạy thử 100 ảnh
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (128, 128))
#         img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # Chuẩn hóa ảnh

#         real_labels = torch.ones(1, 1, 128, 128)
#         fake_labels = torch.zeros(1, 1, 128, 128)

#         # Train Discriminator
#         optimizer_D.zero_grad()
#         real_output = D(img)
#         real_loss = criterion(real_output, real_labels)

#         fake_img = G(img)
#         fake_output = D(fake_img.detach())
#         fake_loss = criterion(fake_output, fake_labels)

#         d_loss = real_loss + fake_loss
#         d_loss.backward()
#         optimizer_D.step()

#         # Train Generator
#         optimizer_G.zero_grad()
#         fake_output = D(fake_img)
#         g_loss = criterion(fake_output, real_labels)
#         g_loss.backward()
#         optimizer_G.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# torch.save(G, "face_swap_generator_full.pth")
# torch.save(D, "face_swap_discriminator_full.pth")
# print("Training và Lưu hoàn tất!")

# ========== Testing =============
# Bước 1: Tải modelmodel
G = torch.load("face_swap_generator_full.pth", weights_only=False)
D = torch.load("face_swap_discriminator_full.pth", weights_only=False)
G.eval()
D.eval()

def swap_face(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    swapped_face = model(img_tensor).detach().squeeze().permute(1, 2, 0).numpy() * 255
    swapped_face = swapped_face.astype(np.uint8)

    cv2.imshow("Swapped Face", swapped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test ảnh bất kỳ
swap_face("CelebA/processed_faces/000007.jpg", G)