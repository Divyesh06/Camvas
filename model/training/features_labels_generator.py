import numpy as np
import os
import cv2
import pickle

import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

classes = ['square', 'circle', 'triangle', 'line', 'unknown']
x_load = []
y_load = []

def process_image(img_path):
    """Process a single image and return its array representation using OpenCV"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
    img_array = img.astype('float32') / 255.
    return img_array.flatten()

def load_class_images(shape_class, class_idx):
    """Load and process images for a single class using threading"""
    image_files = glob.glob(f"data/{shape_class}/*.[pjJgG][pPng]*")
    class_images = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all image processing tasks
        futures = [executor.submit(process_image, img_path) for img_path in image_files]
        
        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), 
                         total=len(image_files), 
                         desc=f"Loading {shape_class}", 
                         unit="img"):
            class_images.append(future.result())
    
    x = np.array(class_images)
    y = np.array([class_idx] * len(class_images)).astype('float32').reshape(-1, 1)
    
    return x, y

def load_image_data():
    """Load image data for all classes using threading"""
    for class_idx, shape_class in enumerate(classes):
        x, y = load_class_images(shape_class, class_idx)
        x_load.append(x)
        y_load.append(y)
    return x_load, y_load

if __name__ == "__main__":
    print("Starting data loading process...")
    features, labels = load_image_data()
    
    # Combine all classes
    features = np.concatenate(x_load).astype('float32')
    labels = np.concatenate(y_load).astype('float32')
    
    print(f"Final features shape: {features.shape}")
    print(f"Final labels shape: {labels.shape}")
    
    # Save the data
    with open("features", "wb") as f:
        pickle.dump(features, f)
    with open("labels", "wb") as f:
        pickle.dump(labels, f)
    
    print("Data saved successfully!")