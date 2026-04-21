import numpy as np
import cv2
import pickle
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

classes = ['square', 'circle', 'triangle', 'line']
x_load = []
y_load = []

IMAGE_SIZE = 28


def process_image(img_path):
    """Load an image, resize to 28x28, and binarize so stroke pixels are 255.

    Binarization makes the training distribution invariant to stroke color,
    which lets inference use any color it wants (the app draws in blue).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return img.flatten().astype(np.uint8)


def load_class_images(shape_class, class_idx):
    """Load and process images for a single class using threading"""
    image_files = glob.glob(f"ml/data/{shape_class}/*.[pjJgG][pPng]*")
    class_images = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in image_files]

        for future in tqdm(as_completed(futures),
                         total=len(image_files),
                         desc=f"Loading {shape_class}",
                         unit="img"):
            class_images.append(future.result())

    x = np.array(class_images, dtype=np.uint8)
    y = np.full((len(class_images), 1), class_idx, dtype=np.int32)

    return x, y


def load_image_data():
    """Load image data for all classes using threading"""
    for class_idx, shape_class in enumerate(classes):
        x, y = load_class_images(shape_class, class_idx)
        x_load.append(x)
        y_load.append(y)
    return x_load, y_load


if __name__ == "__main__":

    features, labels = load_image_data()

    features = np.concatenate(x_load).astype(np.uint8)
    labels = np.concatenate(y_load).astype(np.int32)

    print(f"Final features shape: {features.shape}, dtype: {features.dtype}")
    print(f"Final labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"Class distribution: {np.bincount(labels.flatten())}")

    with open("ml/models/features", "wb") as f:
        pickle.dump(features, f)
    with open("ml/models/labels", "wb") as f:
        pickle.dump(labels, f)

    print("Data saved successfully!")
