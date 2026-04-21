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
    """Hand-drawn arrow with a deliberately messy, spread-out head.

    Real users rarely draw clean 30-degree barbs. This randomises the barb
    angles, lengths (often asymmetric), thickness, and occasionally adds
    filled heads, scribbled tips, or a shaft that slightly overshoots the
    tip.
    """
    x2, y2 = random.randint(38, 90), random.randint(38, 90)
    angle = random.uniform(0, 2 * np.pi)
    length = random.randint(38, 68)
    dx = int(np.cos(angle) * length)
    dy = int(np.sin(angle) * length)
    start = (x2 - dx, y2 - dy)
    end = (x2, y2)

    shaft_thickness = _random_thickness()
    shaft_wobble = random.uniform(0.4, 1.2)

    # Occasionally the shaft overshoots the tip (real drawings do this).
    overshoot = random.uniform(0, 6) if random.random() < 0.35 else 0.0
    shaft_end = (int(end[0] + np.cos(angle) * overshoot),
                 int(end[1] + np.sin(angle) * overshoot))
    draw_handdrawn_line(img, start, shaft_end,
                        thickness=shaft_thickness, wobble=shaft_wobble)

    # Barb angles are asymmetric and much more spread than 30 degrees.
    barb_angle_a = random.uniform(np.pi / 7, np.pi / 2.4)   # ~26-75 deg
    barb_angle_b = random.uniform(np.pi / 7, np.pi / 2.4)
    # Barb lengths are a meaningful fraction of the shaft, often asymmetric.
    barb_len_a = random.uniform(0.25, 0.55) * length
    barb_len_b = random.uniform(0.25, 0.55) * length

    a1 = angle + np.pi + barb_angle_a  # pointing backward from the tip
    a2 = angle + np.pi - barb_angle_b
    p1 = (int(end[0] + np.cos(a1) * barb_len_a),
          int(end[1] + np.sin(a1) * barb_len_a))
    p2 = (int(end[0] + np.cos(a2) * barb_len_b),
          int(end[1] + np.sin(a2) * barb_len_b))

    head_thickness = random.choice([2, 3, 3, 4])
    head_wobble = random.uniform(0.8, 2.0)  # much messier than shaft

    style = random.random()
    if style < 0.55:
        # Standard two-barb head, hand-drawn with heavier wobble.
        draw_handdrawn_line(img, end, p1, thickness=head_thickness, wobble=head_wobble)
        draw_handdrawn_line(img, end, p2, thickness=head_thickness, wobble=head_wobble)
    elif style < 0.75:
        # Closed triangular head (users often fill/close it in).
        pts = np.array([end, p1, p2], dtype=np.int32)
        if random.random() < 0.5:
            cv2.fillPoly(img, [pts], (255, 255, 255))
        else:
            draw_handdrawn_line(img, end, p1, thickness=head_thickness, wobble=head_wobble)
            draw_handdrawn_line(img, end, p2, thickness=head_thickness, wobble=head_wobble)
            draw_handdrawn_line(img, p1, p2, thickness=head_thickness, wobble=head_wobble)
    elif style < 0.9:
        # Two barbs plus an extra scribble/stroke near the tip (messy tip).
        draw_handdrawn_line(img, end, p1, thickness=head_thickness, wobble=head_wobble)
        draw_handdrawn_line(img, end, p2, thickness=head_thickness, wobble=head_wobble)
        # Extra stroke crossing near the tip — simulates re-tracing the head.
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        draw_handdrawn_line(img, end, mid,
                            thickness=head_thickness,
                            wobble=random.uniform(1.2, 2.5))
    else:
        # Single-barb or very asymmetric head.
        draw_handdrawn_line(img, end, p1, thickness=head_thickness, wobble=head_wobble)
        if random.random() < 0.6:
            # Stub of a second barb
            stub_len = barb_len_b * random.uniform(0.2, 0.5)
            p2_stub = (int(end[0] + np.cos(a2) * stub_len),
                       int(end[1] + np.sin(a2) * stub_len))
            draw_handdrawn_line(img, end, p2_stub,
                                thickness=head_thickness, wobble=head_wobble)


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


def _draw_scribble(img):
    """Multiple overlapping hand-drawn strokes in random directions."""
    n_strokes = random.randint(2, 5)
    for _ in range(n_strokes):
        x1, y1 = random.randint(20, 108), random.randint(20, 108)
        x2, y2 = random.randint(20, 108), random.randint(20, 108)
        draw_handdrawn_line(img, (x1, y1), (x2, y2),
                            thickness=_random_thickness(),
                            wobble=random.uniform(1.5, 3.0))


def _draw_zigzag(img):
    """Zigzag across the canvas — the model might confuse this with a line."""
    n_points = random.randint(4, 8)
    x_vals = np.linspace(20, 108, n_points).astype(int)
    y_base = random.randint(40, 88)
    amplitude = random.randint(10, 30)
    pts = []
    for i, x in enumerate(x_vals):
        y = y_base + (amplitude if i % 2 == 0 else -amplitude) + random.randint(-4, 4)
        pts.append((int(x + random.randint(-3, 3)), int(y)))
    angle = random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = CANVAS_SIZE // 2, CANVAS_SIZE // 2
    rot_pts = []
    for x, y in pts:
        rx = int((x - cx) * cos_a - (y - cy) * sin_a + cx)
        ry = int((x - cx) * sin_a + (y - cy) * cos_a + cy)
        rot_pts.append((rx, ry))
    thickness = _random_thickness()
    for i in range(len(rot_pts) - 1):
        draw_handdrawn_line(img, rot_pts[i], rot_pts[i + 1],
                            thickness=thickness,
                            wobble=random.uniform(0.6, 1.2))


def _draw_spiral(img):
    """Outward spiral — smooth curve that isn't a circle or line."""
    cx, cy = random.randint(50, 78), random.randint(50, 78)
    turns = random.uniform(1.5, 3.5)
    max_r = random.randint(20, 40)
    n_points = 80
    points = []
    for i in range(n_points):
        t = i / (n_points - 1)
        r = max_r * t
        theta = turns * 2 * np.pi * t
        x = int(cx + r * np.cos(theta) + np.random.normal(0, 0.8))
        y = int(cy + r * np.sin(theta) + np.random.normal(0, 0.8))
        points.append((x, y))
    thickness = _random_thickness()
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def _draw_arc(img):
    """Open arc — partial circle that shouldn't be classified as a circle."""
    cx, cy = random.randint(45, 83), random.randint(45, 83)
    radius = random.randint(25, 42)
    start_deg = random.uniform(0, 360)
    sweep = random.uniform(90, 260)
    n = 40
    points = []
    for i in range(n):
        theta = np.deg2rad(start_deg + sweep * i / (n - 1))
        x = int(cx + radius * np.cos(theta) + np.random.normal(0, 1.0))
        y = int(cy + radius * np.sin(theta) + np.random.normal(0, 1.0))
        points.append((x, y))
    thickness = _random_thickness()
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def _draw_angle(img):
    """Two lines meeting at a vertex — V or L shape, not a closed triangle."""
    vx, vy = random.randint(40, 88), random.randint(40, 88)
    a1 = random.uniform(0, 2 * np.pi)
    a2 = a1 + random.uniform(np.pi / 6, 5 * np.pi / 6) * random.choice([-1, 1])
    len1 = random.randint(30, 55)
    len2 = random.randint(30, 55)
    p1 = (int(vx + np.cos(a1) * len1), int(vy + np.sin(a1) * len1))
    p2 = (int(vx + np.cos(a2) * len2), int(vy + np.sin(a2) * len2))
    thickness = _random_thickness()
    wobble = random.uniform(0.6, 1.2)
    draw_handdrawn_line(img, (vx, vy), p1, thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, (vx, vy), p2, thickness=thickness, wobble=wobble)


def _draw_cross(img):
    """Plus or X — two crossing lines."""
    cx, cy = random.randint(45, 83), random.randint(45, 83)
    angle = random.uniform(0, np.pi / 2)
    length = random.randint(25, 45)
    dx1, dy1 = int(np.cos(angle) * length), int(np.sin(angle) * length)
    dx2, dy2 = int(-np.sin(angle) * length), int(np.cos(angle) * length)
    thickness = _random_thickness()
    wobble = random.uniform(0.6, 1.2)
    draw_handdrawn_line(img, (cx - dx1, cy - dy1), (cx + dx1, cy + dy1),
                        thickness=thickness, wobble=wobble)
    draw_handdrawn_line(img, (cx - dx2, cy - dy2), (cx + dx2, cy + dy2),
                        thickness=thickness, wobble=wobble)


def _draw_incomplete_shape(img):
    """Partially drawn square/triangle with one side missing."""
    kind = random.choice(["square", "triangle"])
    if kind == "square":
        x, y = random.randint(20, 60), random.randint(20, 60)
        w, h = random.randint(30, 60), random.randint(30, 60)
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        skip = random.randint(0, 3)
        thickness = _random_thickness()
        wobble = random.uniform(0.8, 1.4)
        for i in range(4):
            if i == skip:
                continue
            draw_handdrawn_line(img, corners[i], corners[(i + 1) % 4],
                                thickness=thickness, wobble=wobble)
    else:
        size = random.randint(28, 50)
        base = np.array([[0, -size], [-size * 0.866, size * 0.5], [size * 0.866, size * 0.5]])
        ang = random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        rotated = base @ rot.T
        cx, cy = random.randint(45, 83), random.randint(45, 83)
        pts = [(int(p[0] + cx), int(p[1] + cy)) for p in rotated]
        skip = random.randint(0, 2)
        thickness = _random_thickness()
        wobble = random.uniform(0.8, 1.4)
        edges = [(0, 1), (1, 2), (2, 0)]
        for i, (a, b) in enumerate(edges):
            if i == skip:
                continue
            draw_handdrawn_line(img, pts[a], pts[b], thickness=thickness, wobble=wobble)


def _draw_multi(img):
    """Two small overlapping scribbles/curves — confuses the model into picking one."""
    for _ in range(2):
        cx, cy = random.randint(30, 98), random.randint(30, 98)
        size = random.randint(10, 22)
        n = random.randint(15, 30)
        pts = []
        for i in range(n):
            theta = 2 * np.pi * i / n
            r = size + random.randint(-3, 3)
            x = int(cx + r * np.cos(theta) + np.random.normal(0, 1.0))
            y = int(cy + r * np.sin(theta) + np.random.normal(0, 1.0))
            pts.append((x, y))
        thickness = random.choice([2, 3])
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def _draw_dots(img):
    """Scattered dots / short marks — too sparse to be any shape."""
    n = random.randint(3, 9)
    for _ in range(n):
        cx, cy = random.randint(20, 108), random.randint(20, 108)
        if random.random() < 0.5:
            cv2.circle(img, (cx, cy), random.randint(1, 3), (255, 255, 255), -1)
        else:
            dx, dy = random.randint(-6, 6), random.randint(-6, 6)
            cv2.line(img, (cx, cy), (cx + dx, cy + dy), (255, 255, 255), 2, lineType=cv2.LINE_AA)


def _draw_curve(img):
    """Single random Bezier-like curve."""
    p0 = (random.randint(20, 50), random.randint(20, 108))
    p2 = (random.randint(78, 108), random.randint(20, 108))
    p1 = (random.randint(30, 98), random.randint(20, 108))
    n = 40
    points = []
    for i in range(n):
        t = i / (n - 1)
        x = int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0] + np.random.normal(0, 0.8))
        y = int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1] + np.random.normal(0, 0.8))
        points.append((x, y))
    thickness = _random_thickness()
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


_NONE_GENERATORS = [
    _draw_scribble,
    _draw_zigzag,
    _draw_spiral,
    _draw_arc,
    _draw_angle,
    _draw_cross,
    _draw_incomplete_shape,
    _draw_multi,
    _draw_dots,
    _draw_curve,
]


def draw_none(img):
    """Dispatch to one of the negative-sample generators at random."""
    random.choice(_NONE_GENERATORS)(img)


SHAPES = {
    "circle": draw_circle,
    "line": draw_line,
    "square": draw_square,
    "triangle": draw_triangle,
    "arrow": draw_arrow,
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
