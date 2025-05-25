import cv2
import numpy as np
import os
import random

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

def draw_handdrawn_line(img, pt1, pt2, thickness=2, color=(255, 255, 255), wobble=0.6):
    points = []
    for i in range(21):
        t = i / 20
        x = int(pt1[0] * (1 - t) + pt2[0] * t + np.random.normal(0, wobble))
        y = int(pt1[1] * (1 - t) + pt2[1] * t + np.random.normal(0, wobble))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness, lineType=cv2.LINE_AA)

def draw_triangle(img, pt1, pt2, pt3):
    draw_handdrawn_line(img, pt1, pt2, wobble=1.2)
    draw_handdrawn_line(img, pt2, pt3, wobble=1.2)
    draw_handdrawn_line(img, pt3, pt1, wobble=1.2)

def generate_triangles(output_dir="data/triangle", num=100000):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num):
        img = np.zeros((128, 128, 3), dtype=np.uint8)

        # Base triangle (centered around origin)
        size = random.randint(30, 50)
        pt1 = np.array([0, -size])
        pt2 = np.array([-size * 0.866, size * 0.5])  # 60 degree
        pt3 = np.array([size * 0.866, size * 0.5])

        # Random rotation
        angle = random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pt1 = rot @ pt1
        pt2 = rot @ pt2
        pt3 = rot @ pt3

        # Translate to random canvas position
        cx, cy = random.randint(50, 78), random.randint(50, 78)
        pt1 = (int(pt1[0] + cx), int(pt1[1] + cy))
        pt2 = (int(pt2[0] + cx), int(pt2[1] + cy))
        pt3 = (int(pt3[0] + cx), int(pt3[1] + cy))

        draw_triangle(img, pt1, pt2, pt3)
        result = crop_and_resize(img)
        cv2.imwrite(f"{output_dir}/triangle_{i}.png", result)

generate_triangles()
