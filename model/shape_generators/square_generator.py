import cv2
import numpy as np
import os
import random

def draw_handdrawn_line(img, pt1, pt2, thickness=2, color=(255, 255, 255), wobble=0.6):
    points = []
    for i in range(21):
        t = i / 20
        x = int(pt1[0] * (1 - t) + pt2[0] * t + np.random.normal(0, wobble))
        y = int(pt1[1] * (1 - t) + pt2[1] * t + np.random.normal(0, wobble))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness, lineType=cv2.LINE_AA)

def draw_rectangle(img, x, y, w, h):
    # Add small random jitter to each corner
    tl = (x + random.randint(-3, 3), y + random.randint(-3, 3))
    tr = (x + w + random.randint(-3, 3), y + random.randint(-3, 3))
    br = (x + w + random.randint(-3, 3), y + h + random.randint(-3, 3))
    bl = (x + random.randint(-3, 3), y + h + random.randint(-3, 3))

    # Draw wiggly edges
    draw_handdrawn_line(img, tl, tr, wobble=1.2)
    draw_handdrawn_line(img, tr, br, wobble=1.2)
    draw_handdrawn_line(img, br, bl, wobble=1.2)
    draw_handdrawn_line(img, bl, tl, wobble=1.2)

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

def generate_rectangles(output_dir="data/square", num=100):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        x, y = random.randint(20, 60), random.randint(20, 60)
        w, h = random.randint(30, 60), random.randint(30, 60)
        draw_rectangle(img, x, y, w, h)
        result = crop_and_resize(img)
        cv2.imwrite(f"{output_dir}/rect_{i}.png", result)

generate_rectangles()
