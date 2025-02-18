import glob
import os

def data_preparation(SAVE_DIR):
    """
    Chuẩn bị danh sách ảnh đã xử lý cho quá trình training.
    """
    processed_images = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))

    if not processed_images:
        print("⚠️ Không tìm thấy ảnh nào trong thư mục!", SAVE_DIR)
        return []

    dataset = [img_path for img_path in processed_images if os.path.getsize(img_path) > 1024]  # Chỉ lấy ảnh hợp lệ

    print(f"✅ Tổng số ảnh hợp lệ trong dataset: {len(dataset)}")

    return dataset