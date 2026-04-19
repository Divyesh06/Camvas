import cv2
import numpy as np
import os
import random

def draw_handdrawn_ellipse(img, center, axes, angle, wobble=1.0, thickness=3):
    points = []
    for i in range(72):
        theta = np.deg2rad(i * 5)
        x = axes[0] * np.cos(theta)
        y = axes[1] * np.sin(theta)

        # Apply rotation
        rot_x = x * np.cos(np.deg2rad(angle)) - y * np.sin(np.deg2rad(angle))
        rot_y = x * np.sin(np.deg2rad(angle)) + y * np.cos(np.deg2rad(angle))

        noisy_x = int(center[0] + rot_x + np.random.normal(0, wobble))
        noisy_y = int(center[1] + rot_y + np.random.normal(0, wobble))
        points.append((noisy_x, noisy_y))

    for i in range(len(points)):
        cv2.line(img, points[i], points[(i+1) % len(points)], (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def crop_and_resize(img, size=(28, 28)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 10).astype(np.uint8))
    if coords is None:
        return np.zeros((*size, 3), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    scale = min(size[0]/w, size[1]/h)
    resized = cv2.resize(cropped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    final = np.zeros((*size, 3), dtype=np.uint8)
    y_offset = (size[1] - resized.shape[0]) // 2
    x_offset = (size[0] - resized.shape[1]) // 2
    final[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    return final

def generate_circles(output_dir="data/circle", num=100):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cx, cy = random.randint(40, 88), random.randint(40, 88)
        ax1 = random.randint(25, 40)  # width radius
        ax2 = random.randint(25, 40) if random.random() > 0.3 else ax1  # height radius (allow oval)
        angle = random.uniform(0, 360)

        draw_handdrawn_ellipse(img, (cx, cy), (ax1, ax2), angle, wobble=1.0)
        result = crop_and_resize(img)
        cv2.imwrite(f"{output_dir}/circle_{i}.png", result)

generate_circles()
