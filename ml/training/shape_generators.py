"""Synthetic hand-drawn shape dataset generators."""

import os
import random

import cv2
import numpy as np

CANVAS_SIZE = 128
OUTPUT_SIZE = 28
DEFAULT_COUNT = 2000
DEFAULT_OUTPUT_DIR = "ml/data"


def draw_handdrawn_line(img, pt1, pt2, thickness=2, color=(255, 255, 255), wobble=0.6):
    points = []
    for i in range(21):
        t = i / 20
        x = int(pt1[0] * (1 - t) + pt2[0] * t + np.random.normal(0, wobble))
        y = int(pt1[1] * (1 - t) + pt2[1] * t + np.random.normal(0, wobble))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness, lineType=cv2.LINE_AA)


def crop_and_resize(img, size=(OUTPUT_SIZE, OUTPUT_SIZE)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 10).astype(np.uint8))
    if coords is None:
        return np.zeros((*size, 3), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]
    scale = min(size[0] / w, size[1] / h)
    resized = cv2.resize(cropped, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    final = np.zeros((*size, 3), dtype=np.uint8)
    y_offset = (size[1] - resized.shape[0]) // 2
    x_offset = (size[0] - resized.shape[1]) // 2
    final[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return final


def _random_thickness():
    return random.choice([2, 2, 3, 3, 4])


def draw_line(img):
    x1, y1 = random.randint(20, 100), random.randint(20, 100)
    angle = random.uniform(0, 2 * np.pi)
    length = random.randint(40, 85)
    x2 = x1 + int(np.cos(angle) * length)
    y2 = y1 + int(np.sin(angle) * length)
    draw_handdrawn_line(img, (x1, y1), (x2, y2), thickness=_random_thickness(), wobble=random.uniform(0.4, 1.0))


def draw_arrow(img):
    x2, y2 = random.randint(40, 88), random.randint(40, 88)
    angle = random.uniform(0, 2 * np.pi)
    length = random.randint(40, 60)
    dx = int(np.cos(angle) * length)
    dy = int(np.sin(angle) * length)
    start = (x2 - dx, y2 - dy)
    end = (x2, y2)
    draw_handdrawn_line(img, start, end, thickness=3, wobble=0.7)

    arrow_size = 15
    angle1 = angle + np.pi / 6
    angle2 = angle - np.pi / 6
    p1 = (int(end[0] - arrow_size * np.cos(angle1)), int(end[1] - arrow_size * np.sin(angle1)))
    p2 = (int(end[0] - arrow_size * np.cos(angle2)), int(end[1] - arrow_size * np.sin(angle2)))
    cv2.line(img, end, p1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, end, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def draw_square(img):
    x, y = random.randint(15, 60), random.randint(15, 60)
    w, h = random.randint(30, 70), random.randint(30, 70)
    jitter = random.randint(2, 5)
    tl = (x + random.randint(-jitter, jitter), y + random.randint(-jitter, jitter))
    tr = (x + w + random.randint(-jitter, jitter), y + random.randint(-jitter, jitter))
    br = (x + w + random.randint(-jitter, jitter), y + h + random.randint(-jitter, jitter))
    bl = (x + random.randint(-jitter, jitter), y + h + random.randint(-jitter, jitter))
    thickness = _random_thickness()
    wobble = random.uniform(0.8, 1.5)
    draw_handdrawn_line(img, tl, tr, thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, tr, br, thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, br, bl, thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, bl, tl, thickness=thickness, wobble=wobble)


def draw_triangle(img):
    size = random.randint(28, 55)
    base = np.array([
        [0, -size],
        [-size * 0.866, size * 0.5],
        [size * 0.866, size * 0.5],
    ])
    angle = random.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = base @ rot.T
    cx, cy = random.randint(45, 83), random.randint(45, 83)
    pts = [(int(p[0] + cx), int(p[1] + cy)) for p in rotated]
    thickness = _random_thickness()
    wobble = random.uniform(0.8, 1.5)
    draw_handdrawn_line(img, pts[0], pts[1], thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, pts[1], pts[2], thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, pts[2], pts[0], thickness=thickness, wobble=wobble)


def draw_circle(img):
    cx, cy = random.randint(40, 88), random.randint(40, 88)
    ax1 = random.randint(22, 42)
    ax2 = random.randint(22, 42) if random.random() > 0.3 else ax1
    angle_deg = random.uniform(0, 360)
    wobble = random.uniform(0.7, 1.3)
    thickness = _random_thickness()
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))
    points = []
    for i in range(72):
        theta = np.deg2rad(i * 5)
        x = ax1 * np.cos(theta)
        y = ax2 * np.sin(theta)
        rot_x = x * cos_a - y * sin_a
        rot_y = x * sin_a + y * cos_a
        noisy_x = int(cx + rot_x + np.random.normal(0, wobble))
        noisy_y = int(cy + rot_y + np.random.normal(0, wobble))
        points.append((noisy_x, noisy_y))
    for i in range(len(points)):
        cv2.line(img, points[i], points[(i + 1) % len(points)], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


SHAPES = {
    "circle": draw_circle,
    "line": draw_line,
    "square": draw_square,
    "triangle": draw_triangle,
}


def generate_all(output_dir=DEFAULT_OUTPUT_DIR, count=DEFAULT_COUNT, shapes=None):
    names = shapes if shapes is not None else list(SHAPES.keys())
    for name in names:

        print("Generating", count, name, "images")
        draw_fn = SHAPES[name]
        shape_dir = os.path.join(output_dir, name)
        os.makedirs(shape_dir, exist_ok=True)
        for i in range(count):
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
            draw_fn(canvas)
            final_img = crop_and_resize(canvas)
            cv2.imwrite(os.path.join(shape_dir, f"{name}_{i}.png"), final_img)


if __name__ == "__main__":

    generate_all()
