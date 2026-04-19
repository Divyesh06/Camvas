import cv2
import numpy as np
import os
import random

def draw_handdrawn_arrow(img, start, end, thickness=3, color=(255, 255, 255), wobble=0.7):
    points = []
    num_points = 20
    for i in range(num_points + 1):
        t = i / num_points
        x = int(start[0] * (1 - t) + end[0] * t + np.random.normal(0, wobble))
        y = int(start[1] * (1 - t) + end[1] * t + np.random.normal(0, wobble))
        points.append((x, y))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness, lineType=cv2.LINE_AA)

    # Arrowhead
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    arrow_size = 15
    angle1 = angle + np.pi / 6
    angle2 = angle - np.pi / 6
    p1 = (int(end[0] - arrow_size * np.cos(angle1)), int(end[1] - arrow_size * np.sin(angle1)))
    p2 = (int(end[0] - arrow_size * np.cos(angle2)), int(end[1] - arrow_size * np.sin(angle2)))
    cv2.line(img, end, p1, color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, end, p2, color, 2, lineType=cv2.LINE_AA)

def crop_and_resize(img, size=(28, 28)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 10).astype(np.uint8))
    if coords is None:
        return np.zeros((*size, 3), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    # Resize with aspect ratio
    h_new, w_new = cropped.shape[:2]
    scale = min(size[0]/w_new, size[1]/h_new)
    resized = cv2.resize(cropped, (int(w_new*scale), int(h_new*scale)), interpolation=cv2.INTER_AREA)

    # Center on black background
    final_img = np.zeros((*size, 3), dtype=np.uint8)
    y_offset = (size[1] - resized.shape[0]) // 2
    x_offset = (size[0] - resized.shape[1]) // 2
    final_img[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    return final_img

def generate_arrow_dataset(output_dir="data/arrow", num_images=100):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        canvas = np.zeros((128, 128, 3), dtype=np.uint8)  # Black background

        # 1. Choose end point
        x2, y2 = random.randint(40, 88), random.randint(40, 88)

        # 2. Choose angle
        angle = random.uniform(0, 2 * np.pi)

        # 3. Generate start point a bit farther (tail longer)
        length = random.randint(40, 60)  # longer than before
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length)
        x1 = x2 - dx
        y1 = y2 - dy

        draw_handdrawn_arrow(canvas, (x1, y1), (x2, y2))

        final_img = crop_and_resize(canvas)
        cv2.imwrite(os.path.join(output_dir, f"arrow_{i}.png"), final_img)

generate_arrow_dataset()
