import glob
import os

def data_preparation(SAVE_DIR):
    processed_images = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))
    dataset = []

    for img_path in processed_images:
        dataset.append(img_path)

    print(f"Tổng số ảnh trong dataset: {len(dataset)}")
    return dataset