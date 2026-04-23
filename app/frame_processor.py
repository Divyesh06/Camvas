import cv2
import numpy as np
import time
import sys
import onnxruntime as ort
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


def is_frozen():
    return getattr(sys, 'frozen', False)

if not is_frozen():
    model_path = "ml/models/ShapeDetection.onnx"
    hand_model_path = "ml/models/hand_landmarker.task"
else:
    app_dir = os.path.dirname(sys.executable)
    model_path = os.path.join(app_dir, 'models', 'ShapeDetection.onnx')
    hand_model_path = os.path.join(app_dir, 'models', 'hand_landmarker.task')

onnx_session = ort.InferenceSession(model_path)
input_name = onnx_session.get_inputs()[0].name

THUMB_TIP = 4
INDEX_FINGER_TIP = 8



_hand_detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
)
_last_video_ts_ms = -1

# Global variables
canvas = None
canvas_dirty = False
permanent_canvas = None
undo = []
redo = []
init = False
drawing = False
thickness = None
last_x = None
last_y = None
draw_cooldown_threshold = 0
undo_cooldown_threshold = 0
last_time = time.time()

# New variables for selection tool
tool = 'draw'  # Default tool is 'draw', can be 'select'
shapes = []  # List to store all shapes
selected_shape_idx = None  # Index of currently selected shape
is_moving_shape = False  # Flag to track if we're currently moving a shape
move_start_pos = None  # Starting position for shape movement

_frame_counter = 0
_last_results = []
_buttons = []
_pending_undo = False
_pending_tool_toggle = False
_pending_clear_all = False
_pending_color_cycle = False

UNDO_LIMIT = 20

# Drawing colours (BGR). Pure black would be invisible against the
# additive blend into `frame` (and would drop out of the non-zero scan in
# find_extreme_non_zero_points), so "black" is a near-black that renders
# as black but registers as a stroke.
DRAW_COLORS = [
    ("red",    (0, 0, 255)),
    ("white",  (255, 255, 255)),
    ("black",  (20, 20, 20)),
    ("blue",   (255, 0, 0)),
    ("green",  (0, 200, 0)),
    ("yellow", (0, 220, 220)),
   
]
_color_idx = 0


def get_draw_color():
    return DRAW_COLORS[_color_idx][1]


def _cycle_color():
    global _color_idx
    _color_idx = (_color_idx + 1) % len(DRAW_COLORS)

# Exponential smoothing factor for drawn strokes. Each new point is blended
# against the last one as `new = a * raw + (1 - a) * last`. Lower values =
# smoother strokes at the cost of more lag behind the finger.
STROKE_SMOOTH_ALPHA = 0.5

# Opacity of strokes when composited onto the video frame. 1.0 = fully
# opaque replacement, 0.5 = half-transparent blend, etc. Lower values
# will let the video show through.
STROKE_OPACITY = 0.9


def _request_undo():
    """Signal process_frame to run undo on its next tick.

    keyboard's hotkey handler runs on a background thread; mutating
    permanent_canvas/shapes/undo/redo directly from here would race with the
    main frame callback. Flipping a flag and consuming it from process_frame
    keeps all state changes on one thread.
    """
    global _pending_undo
    _pending_undo = True


def _request_tool_toggle():
    global _pending_tool_toggle
    _pending_tool_toggle = True


def _request_clear_all():
    global _pending_clear_all
    _pending_clear_all = True


def _request_color_cycle():
    global _pending_color_cycle
    _pending_color_cycle = True


try:
    import keyboard as _keyboard
    _keyboard.add_hotkey('ctrl+c+z', _request_undo, suppress=False)
    _keyboard.add_hotkey('ctrl+c+s', _request_tool_toggle, suppress=False)
    _keyboard.add_hotkey('ctrl+c+x', _request_clear_all, suppress=False)
    _keyboard.add_hotkey('ctrl+c+v', _request_color_cycle, suppress=False)
except Exception as _e:
    print(f"Keyboard hotkeys unavailable: {_e}")


def _bgra(color, alpha):
    a = int(round(max(0.0, min(1.0, alpha)) * 255))
    return (color[0], color[1], color[2], a)


def _draw_pill(target, center, width, height, color, alpha):
    """Paint a rounded-rect pill onto a BGRA target at the given alpha."""
    r = height // 2
    x1 = center[0] - width // 2
    y1 = center[1] - height // 2
    x2 = x1 + width
    y2 = y1 + height
    bgra = _bgra(color, alpha)
    cv2.rectangle(target, (x1 + r, y1), (x2 - r, y2), bgra, -1, lineType=cv2.LINE_AA)
    cv2.circle(target, (x1 + r, center[1]), r, bgra, -1, lineType=cv2.LINE_AA)
    cv2.circle(target, (x2 - r, center[1]), r, bgra, -1, lineType=cv2.LINE_AA)


def _overlay_strokes(frame, layer, opacity):
    """Overlay a BGR stroke layer onto a BGR frame at the given opacity.

    opacity=1.0 replaces frame pixels with layer pixels (fully opaque).
    0 < opacity < 1 alpha-blends. Only non-zero-colour pixels in `layer`
    are touched, so the rest of the video is untouched.
    """
    if opacity <= 0:
        return
    mask = layer.any(axis=2)
    if not mask.any():
        return
    if opacity >= 1.0:
        frame[mask] = layer[mask]
        return
    a = float(opacity)
    lp = layer[mask].astype(np.float32)
    fp = frame[mask].astype(np.float32)
    frame[mask] = (a * lp + (1 - a) * fp).astype(np.uint8)


def _composite_bgra(frame_bgr, ui_bgra):
    """Alpha-blend a BGRA layer onto a BGR region, in place. Same shape."""
    alpha = ui_bgra[..., 3]
    mask = alpha > 0
    if not mask.any():
        return
    a = alpha[mask].astype(np.float32) / 255.0
    a = a[:, None]
    fb = frame_bgr[mask].astype(np.float32)
    ub = ui_bgra[mask, :3].astype(np.float32)
    frame_bgr[mask] = (a * ub + (1 - a) * fb).astype(np.uint8)


def _interp_bg(progress):
    """Translucent button background — near-white at rest, dark when held."""
    t = max(0.0, min(1.0, progress))
    gray = int(round(240 * (1 - t) + 30 * t))
    color = (gray, gray, gray)
    alpha = 0.30 + 0.55 * t
    return color, alpha


def _foreground_color(progress):
    """Icon/text color; flips from dark (on light bg) to light (on dark bg)."""
    t = max(0.0, min(1.0, progress))
    if t < 0.5:
        return (35, 35, 35)
    return (240, 240, 240)


def _draw_shortcut_pill(target, center, text):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.4
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 4
    width = tw + 2 * pad_x
    height = th + baseline + 2 * pad_y
    _draw_pill(target, center, width, height, (10, 10, 10), 0.55)
    org = (center[0] - tw // 2, center[1] + th // 2 - 1)
    cv2.putText(target, text, org, font, scale, _bgra((240, 240, 240), 1.0),
                thickness, cv2.LINE_AA)


class HoldButton:
    """Pill-shaped button activated by holding a pinch over it.

    Reusable for any hold-to-trigger action. The caller is expected to call
    `tick(dt, pressed)` and `draw(target)` once per frame, plus `contains(point)`
    for hit-testing. The background starts translucent white and darkens as
    progress grows; once progress hits 1.0 the activation callback fires and
    a latch blocks re-firing until the user releases the pinch.
    """

    def __init__(self, center, width, height, hold_seconds, on_activate,
                 icon=None, label=None, shortcut=None, grace_seconds=0.25):
        self.center = center
        self.width = width
        self.height = height
        self.hold_seconds = max(hold_seconds, 0.01)
        self.grace_seconds = max(grace_seconds, 0.0)
        self.on_activate = on_activate
        self.icon = icon
        self.label = label
        self.shortcut = shortcut
        self.progress = 0.0
        self._fired_latch = False
        self._unpressed_time = 0.0

    def contains(self, point):
        dx = abs(point[0] - self.center[0])
        dy = abs(point[1] - self.center[1])
        return dx <= self.width // 2 and dy <= self.height // 2

    def tick(self, dt, pressed):
        if pressed:
            self._unpressed_time = 0.0
            if not self._fired_latch:
                self.progress = min(1.0, self.progress + dt / self.hold_seconds)
                if self.progress >= 1.0:
                    self._fired_latch = True
                    self.on_activate()
            return

        self._unpressed_time += dt
        if self._unpressed_time < self.grace_seconds:
            return
        self.progress = max(0.0, self.progress - dt / self.hold_seconds)
        self._fired_latch = False

    def draw(self, target):
        color, alpha = _interp_bg(self.progress)
        _draw_pill(target, self.center, self.width, self.height, color, alpha)

        fg = _bgra(_foreground_color(self.progress), 1.0)
        icon_size = self.height - 22
        if self.label:
            icon_cx = self.center[0] - self.width // 2 + icon_size // 2 + 14
        else:
            # Icon-only button: center the icon in the pill.
            icon_cx = self.center[0]
        icon_cy = self.center[1]
        if self.icon is not None:
            self.icon(target, (icon_cx, icon_cy), icon_size, fg)

        if self.label:
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(self.label, font, scale, thickness)
            text_x = icon_cx + icon_size // 2 + 10
            text_y = self.center[1] + th // 2 - 1
            cv2.putText(target, self.label, (text_x, text_y),
                        font, scale, fg, thickness, cv2.LINE_AA)

        if self.shortcut:
            shortcut_center = (self.center[0],
                               self.center[1] - self.height // 2 - 16)
            _draw_shortcut_pill(target, shortcut_center, self.shortcut)


class ToggleButton(HoldButton):
    """Two-option horizontal toggle that visibly conveys the active state.

    `states` is a list of (label, icon) pairs; `state_fn()` returns the label
    of the currently-active state. Activation fires `on_activate` which is
    expected to switch the underlying state.
    """

    def __init__(self, *, center, width, height, hold_seconds, on_activate,
                 states, state_fn, shortcut=None, grace_seconds=0.25):
        super().__init__(center=center, width=width, height=height,
                         hold_seconds=hold_seconds, on_activate=on_activate,
                         icon=None, label=None, shortcut=shortcut,
                         grace_seconds=grace_seconds)
        self.states = states
        self.state_fn = state_fn

    def draw(self, target):
        bg_color, bg_alpha = _interp_bg(self.progress)
        _draw_pill(target, self.center, self.width, self.height, bg_color, bg_alpha)

        current = (self.state_fn() or "").lower()
        active_idx = 0
        for i, (lbl, _) in enumerate(self.states):
            if lbl.lower() == current:
                active_idx = i
                break

        half_w = self.width // 2
        inner_pad = 4
        knob_w = half_w - inner_pad
        knob_h = self.height - 2 * inner_pad

        active_cx = (self.center[0] - self.width // 2
                     + half_w * active_idx + half_w // 2)
        _draw_pill(target, (active_cx, self.center[1]),
                   knob_w, knob_h, (30, 30, 30), 0.9)

        icon_size = knob_h - 18
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.5
        thickness = 1
        gap = 6
        for i, (lbl, icon) in enumerate(self.states):
            half_cx = (self.center[0] - self.width // 2
                       + half_w * i + half_w // 2)
            is_active = (i == active_idx)
            fg_color = (245, 245, 245) if is_active else (60, 60, 60)
            fg = _bgra(fg_color, 1.0)

            # Center the icon + gap + text block inside the half so there's
            # no extra padding after the label (fixes overflow on "Select").
            (tw, th), _ = cv2.getTextSize(lbl, font, scale, thickness)
            block_w = icon_size + gap + tw
            block_left = half_cx - block_w // 2
            icon_cx = block_left + icon_size // 2
            icon_cy = self.center[1]
            if icon is not None:
                icon(target, (icon_cx, icon_cy), icon_size, fg)

            text_x = block_left + icon_size + gap
            text_y = self.center[1] + th // 2 - 1
            cv2.putText(target, lbl, (text_x, text_y),
                        font, scale, fg, thickness, cv2.LINE_AA)

        if self.shortcut:
            shortcut_center = (self.center[0],
                               self.center[1] - self.height // 2 - 16)
            _draw_shortcut_pill(target, shortcut_center, self.shortcut)


def _undo_icon(target, center, size, color=(240, 240, 240)):
    r = size // 2
    cv2.ellipse(target, center, (r, r), 0, 40, 320, color, 2, lineType=cv2.LINE_AA)
    tip_angle = np.deg2rad(40)
    tip = (int(center[0] + r * np.cos(tip_angle)),
           int(center[1] + r * np.sin(tip_angle)))
    barb = max(4, r // 2)
    cv2.line(target, tip, (tip[0] - barb, tip[1] - 1),
             color, 2, lineType=cv2.LINE_AA)
    cv2.line(target, tip, (tip[0] + 1, tip[1] + barb),
             color, 2, lineType=cv2.LINE_AA)


def _clear_icon(target, center, size, color=(240, 240, 240)):
    s = size // 2 - 1
    cx, cy = center
    cv2.line(target, (cx - s, cy - s), (cx + s, cy + s),
             color, 2, lineType=cv2.LINE_AA)
    cv2.line(target, (cx - s, cy + s), (cx + s, cy - s),
             color, 2, lineType=cv2.LINE_AA)


def _color_icon(target, center, size, _fg=None):
    """Filled circle in the currently selected drawing colour.

    Ignores the passed `_fg` (button foreground) — the whole point of this
    icon is to reflect the selected draw colour.
    """
    r = size // 2 - 1
    swatch = (*get_draw_color(), 255)
    cv2.circle(target, center, r, swatch, -1, lineType=cv2.LINE_AA)
    # Dark ring so white/yellow swatches stay visible on light button bg.
    cv2.circle(target, center, r, (30, 30, 30, 255), 1, lineType=cv2.LINE_AA)


def _pencil_icon(target, center, size, color=(240, 240, 240)):
    """Pencil at 45°: thick shaft with a filled pointed tip."""
    s = size // 2
    cx, cy = center
    # Shaft: thick diagonal line from tip base to eraser end.
    shaft_thickness = max(3, int(s * 0.45))
    shaft_start = (cx - int(s * 0.4), cy + int(s * 0.4))
    shaft_end = (cx + int(s * 0.9), cy - int(s * 0.9))
    cv2.line(target, shaft_start, shaft_end, color,
             shaft_thickness, lineType=cv2.LINE_AA)
    # Tip: small filled triangle extending past the shaft start.
    tip_pts = np.array([
        [cx - s, cy + s],                             # outermost point
        [cx - int(s * 0.15), cy + int(s * 0.6)],      # upper side
        [cx - int(s * 0.6), cy + int(s * 0.15)],      # lower side
    ], dtype=np.int32)
    cv2.fillPoly(target, [tip_pts], color, lineType=cv2.LINE_AA)


def _cursor_icon(target, center, size, color=(240, 240, 240)):
    """Modern triangular cursor pointing up-left."""
    s = size // 2
    cx, cy = center
    pts = np.array([
        [cx - s, cy - s],                          # tip (up-left)
        [cx + int(s * 0.55), cy + int(s * 0.1)],   # right edge
        [cx - int(s * 0.1), cy + int(s * 0.55)],   # bottom edge
    ], dtype=np.int32)
    cv2.fillPoly(target, [pts], color, lineType=cv2.LINE_AA)


def _push_undo():
    undo.append(permanent_canvas.copy())
    if len(undo) > UNDO_LIMIT:
        del undo[0]


def init_state(frame_shape):
    """Initializes or resets all global variables to a clean state."""
    global canvas, canvas_dirty, permanent_canvas, undo, redo
    global init, drawing, last_x, last_y
    global draw_cooldown_threshold, undo_cooldown_threshold
    global tool, shapes, selected_shape_idx, is_moving_shape, move_start_pos
    global _frame_counter, _last_results

    canvas = np.zeros(frame_shape, dtype=np.uint8)
    canvas_dirty = False
    permanent_canvas = np.zeros(frame_shape, dtype=np.uint8)
    undo = [permanent_canvas.copy()]
    redo = []
    init = True
    drawing = False
    last_x = None
    last_y = None
    draw_cooldown_threshold = 0
    undo_cooldown_threshold = 0
    
    # Initialize selection tool variables
    tool = 'draw'
    shapes = []
    selected_shape_idx = None
    is_moving_shape = False
    move_start_pos = None

    _frame_counter = 0
    _last_results = []

    _init_buttons(frame_shape)


def _init_buttons(frame_shape):
    h, w = frame_shape[:2]
    button_h = 42
    single_w = 130
    toggle_w = 180
    cy = 50 + button_h // 2  # pushed down from the top
    gap = 20

    widths = [single_w, toggle_w, single_w]
    total_w = sum(widths) + gap * (len(widths) - 1)
    start_x = (w - total_w) // 2

    centers = []
    x = start_x
    for bw in widths:
        centers.append((x + bw // 2, cy))
        x += bw + gap

    _buttons.clear()
    # Colour-cycle button: icon-only pill in the top-left corner.
    color_w = 60
    color_margin = 24
    _buttons.append(HoldButton(
        center=(color_margin + color_w // 2, cy),
        width=color_w, height=button_h,
        hold_seconds=0.5, on_activate=_cycle_color,
        icon=_color_icon, shortcut="Ctrl+C+V",
    ))
    _buttons.append(HoldButton(
        center=centers[0], width=single_w, height=button_h,
        hold_seconds=0.5, on_activate=undo_canvas,
        icon=_undo_icon, label="Undo", shortcut="Ctrl+C+Z",
    ))
    _buttons.append(ToggleButton(
        center=centers[1], width=toggle_w, height=button_h,
        hold_seconds=0.5, on_activate=_toggle_tool,
        states=[("Draw", _pencil_icon), ("Select", _cursor_icon)],
        state_fn=lambda: tool,
        shortcut="Ctrl+C+S",
    ))
    _buttons.append(HoldButton(
        center=centers[2], width=single_w, height=button_h,
        hold_seconds=0.5, on_activate=clear_all,
        icon=_clear_icon, label="Clear", shortcut="Ctrl+C+X",
    ))


def index_thumb_close(hand_landmarks):
    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks[THUMB_TIP]
    distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
    return distance < 0.04



def keras_predict(image):
    """Runs prediction Model on the temporary canvas"""

    processed = keras_process_image(image).astype(np.float32)
   
    pred_probab = onnx_session.run(None, {input_name: processed})[0][0]
    pred_class = int(np.argmax(pred_probab))
    
    shape_classes = ['square', 'circle', 'triangle', 'line', 'arrow']
    confidence = float(np.max(pred_probab))

    print(f"I am {confidence * 100:.2f}% sure that this is a {shape_classes[pred_class]}")
    return confidence, pred_class

def correct_image(image, shape):
    global permanent_canvas, shapes
    extremes = find_extreme_non_zero_points(image)
    if extremes is None:
        # Canvas was marked dirty but ended up with no visible pixels (e.g.
        # strokes clipped out of frame). The caller clears canvas/dirty
        # immediately after this returns, so just bail out.
        return
    leftmost, topmost, rightmost, bottommost, center = extremes
    draw_color = get_draw_color()

    # Create a new shape and store its properties
    new_shape = {
        'type': shape,
        'points': [],
        'center': center,
        'bbox': (leftmost[0], topmost[1], rightmost[0], bottommost[1]),  # x1, y1, x2, y2
        'color': draw_color,
    }

    if shape == 'circle':
        radius = center[0] - leftmost[0]
        cv2.circle(permanent_canvas, center, radius, draw_color, 5)
        new_shape['points'] = [center, radius]
    elif shape == 'square':
        cv2.rectangle(permanent_canvas, (leftmost[0], topmost[1]), (rightmost[0], bottommost[1]), draw_color, 5)
        new_shape['points'] = [(leftmost[0], topmost[1]), (rightmost[0], bottommost[1])]
    elif shape == 'triangle':
        if abs(bottommost[1] - min(leftmost[1], rightmost[1])) > abs(topmost[1] - min(leftmost[1], rightmost[1])):
            topmost = bottommost
        cv2.line(permanent_canvas, leftmost, topmost, draw_color, 5)
        cv2.line(permanent_canvas, rightmost, topmost, draw_color, 5)
        cv2.line(permanent_canvas, leftmost, rightmost, draw_color, 5)
        new_shape['points'] = [leftmost, topmost, rightmost]
    elif shape == 'line':
        cv2.line(permanent_canvas, leftmost, rightmost, draw_color, 5)
        new_shape['points'] = [leftmost, rightmost]
    elif shape == 'arrow':
        tail, head = find_arrow_endpoints(image)
        cv2.arrowedLine(permanent_canvas, tail, head, draw_color, 5, tipLength=0.3)
        new_shape['points'] = [tail, head]

    # Add shape to our list of shapes
    shapes.append(new_shape)
    
    _push_undo()
    redo.clear()

def find_extreme_non_zero_points(img):
    non_zero_coords = np.argwhere(img != 0)
    if len(non_zero_coords) == 0:
        return None
    y_coords = non_zero_coords[:, 0]
    x_coords = non_zero_coords[:, 1]

    leftmost = (x_coords.min(), y_coords[x_coords.argmin()])
    rightmost = (x_coords.max(), y_coords[x_coords.argmax()])
    topmost = (x_coords[y_coords.argmin()], y_coords.min())
    bottommost = (x_coords[y_coords.argmax()], y_coords.max())
    center = ((leftmost[0] + rightmost[0]) // 2, (leftmost[1] + rightmost[1]) // 2)

    return leftmost, topmost, rightmost, bottommost, center


def find_arrow_endpoints(image):
    """Find tail/head of a drawn arrow.

    Project all stroke pixels onto the principal axis via PCA, take the two
    extremes as the endpoints, then decide which is the head by comparing
    local pixel density — the arrowhead's barbs make that end denser.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ys, xs = np.nonzero(gray)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    proj = centered @ axis
    end_a = tuple(pts[np.argmin(proj)].astype(int))
    end_b = tuple(pts[np.argmax(proj)].astype(int))

    radius_sq = 15 * 15
    density_a = np.sum((xs - end_a[0]) ** 2 + (ys - end_a[1]) ** 2 < radius_sq)
    density_b = np.sum((xs - end_b[0]) ** 2 + (ys - end_b[1]) ** 2 < radius_sq)
    if density_a > density_b:
        return end_b, end_a
    return end_a, end_b

def keras_process_image(img):
    """Match training preprocessing: binarize, aspect-preserving resize,
    center-pad to 28x28, normalize to [0, 1].

    Binarization makes the model invariant to the blue drawing color — any
    non-zero pixel becomes a stroke pixel, the same representation training saw.
    """
    target = 28
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((1, target, target, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    scale = min(target / w, target / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target, target), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    normalized = padded.astype(np.float32) / 255.0
    return normalized.reshape(1, target, target, 1)

def undo_canvas():
    global permanent_canvas, shapes
    if len(undo) > 1:
        redo.append(undo.pop())
        permanent_canvas = undo[-1].copy()
        if shapes:
            shapes.pop()  # Remove the last shape

def redo_canvas():
    global permanent_canvas
    if redo:
        undo.append(redo.pop())
        permanent_canvas = undo[-1].copy()
        # Note: We would need to store and restore shape info for proper redo


def clear_all():
    """Wipe every drawing layer and reset history to a fresh blank state."""
    global permanent_canvas, canvas, canvas_dirty
    global selected_shape_idx, is_moving_shape, move_start_pos
    permanent_canvas[:] = 0
    canvas[:] = 0
    canvas_dirty = False
    shapes.clear()
    undo.clear()
    undo.append(permanent_canvas.copy())
    redo.clear()
    # Reset selection so the select-tool highlight block doesn't index a
    # now-empty shapes list on the next frame.
    selected_shape_idx = None
    is_moving_shape = False
    move_start_pos = None


def _toggle_tool():
    set_tool('select' if tool == 'draw' else 'draw')


def is_point_in_shape(point, shape):
    """Check if a point is inside or near a shape"""
    x, y = point
    shape_type = shape['type']
    
    if shape_type == 'circle':
        center, radius = shape['points']
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        return distance <= radius + 10  # Add a small margin for easier selection
        
    elif shape_type == 'square':
        (x1, y1), (x2, y2) = shape['points']
        return x1 - 10 <= x <= x2 + 10 and y1 - 10 <= y <= y2 + 10
        
    elif shape_type == 'triangle':
        # Simple bounding box check for triangle
        leftmost, topmost, rightmost = shape['points']
        x_min = min(leftmost[0], topmost[0], rightmost[0]) - 10
        x_max = max(leftmost[0], topmost[0], rightmost[0]) + 10
        y_min = min(leftmost[1], topmost[1], rightmost[1]) - 10
        y_max = max(leftmost[1], topmost[1], rightmost[1]) + 10
        return x_min <= x <= x_max and y_min <= y <= y_max
        
    elif shape_type in ('line', 'arrow'):
        start, end = shape['points']
        # Calculate distance from point to line
        line_length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        if line_length == 0:
            return False

        # Distance from point to line formula
        dist = abs((end[1] - start[1]) * x - (end[0] - start[0]) * y + end[0] * start[1] - end[1] * start[0]) / line_length
        return dist < 15  # Thickness + margin

    return False

def select_shape(point):
    """Try to select a shape at the given point"""
    global selected_shape_idx
    
    # First deselect any currently selected shape
    selected_shape_idx = None
    
    # Check each shape from last to first (top to bottom in z-order)
    for i in range(len(shapes) - 1, -1, -1):
        if is_point_in_shape(point, shapes[i]):
            selected_shape_idx = i
            return True
            
    return False

def move_shape(shape_idx, offset):
    """Move a shape by the given offset"""
    global permanent_canvas, shapes
    
    if shape_idx is None or shape_idx >= len(shapes):
        return
        
    shape = shapes[shape_idx]
    dx, dy = offset
    
    # Create a clean canvas without the selected shape
    temp_canvas = permanent_canvas.copy()
    
    # Update shape coordinates based on its type
    if shape['type'] == 'circle':
        center, radius = shape['points']
        new_center = (center[0] + dx, center[1] + dy)
        shape['points'] = [new_center, radius]
        shape['center'] = new_center
        
    elif shape['type'] == 'square':
        (x1, y1), (x2, y2) = shape['points']
        shape['points'] = [(x1 + dx, y1 + dy), (x2 + dx, y2 + dy)]
        
    elif shape['type'] == 'triangle':
        p1, p2, p3 = shape['points']
        shape['points'] = [(p1[0] + dx, p1[1] + dy), 
                          (p2[0] + dx, p2[1] + dy), 
                          (p3[0] + dx, p3[1] + dy)]
        
    elif shape['type'] in ('line', 'arrow'):
        p1, p2 = shape['points']
        shape['points'] = [(p1[0] + dx, p1[1] + dy),
                          (p2[0] + dx, p2[1] + dy)]

    # Update center and bounding box
    x_coords = [p[0] for p in shape['points'] if isinstance(p, tuple)]
    y_coords = [p[1] for p in shape['points'] if isinstance(p, tuple)]
    
    if x_coords and y_coords:  # Make sure we have valid coordinates
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        shape['center'] = (center_x, center_y)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        shape['bbox'] = (x_min, y_min, x_max, y_max)
    
    # Redraw all shapes on the clean canvas
    redraw_all_shapes()
    
    _push_undo()

def redraw_all_shapes():
    """Redraw all shapes on the permanent canvas"""
    global permanent_canvas, shapes
    
    # Clear the canvas
    permanent_canvas = np.zeros_like(permanent_canvas)
    
    # Redraw each shape in its original colour.
    for shape in shapes:
        color = shape.get('color', (255, 0, 0))
        if shape['type'] == 'circle':
            center, radius = shape['points']
            cv2.circle(permanent_canvas, center, radius, color, 5)

        elif shape['type'] == 'square':
            (x1, y1), (x2, y2) = shape['points']
            cv2.rectangle(permanent_canvas, (x1, y1), (x2, y2), color, 5)

        elif shape['type'] == 'triangle':
            p1, p2, p3 = shape['points']
            cv2.line(permanent_canvas, p1, p2, color, 5)
            cv2.line(permanent_canvas, p2, p3, color, 5)
            cv2.line(permanent_canvas, p3, p1, color, 5)

        elif shape['type'] == 'line':
            p1, p2 = shape['points']
            cv2.line(permanent_canvas, p1, p2, color, 5)

        elif shape['type'] == 'arrow':
            tail, head = shape['points']
            cv2.arrowedLine(permanent_canvas, tail, head, color, 5, tipLength=0.3)

def process_frame(frame, flip=False):
    global canvas, canvas_dirty, last_time, draw_cooldown_threshold, thickness
    global init, drawing, permanent_canvas, last_x, last_y
    global tool, selected_shape_idx, is_moving_shape, move_start_pos
    global _frame_counter, _last_results, _last_video_ts_ms
    global _pending_undo, _pending_tool_toggle, _pending_clear_all
    global _pending_color_cycle

    if not init:
        init_state(frame.shape)

    if _pending_undo:
        _pending_undo = False
        undo_canvas()

    if _pending_tool_toggle:
        _pending_tool_toggle = False
        _toggle_tool()

    if _pending_clear_all:
        _pending_clear_all = False
        clear_all()

    if _pending_color_cycle:
        _pending_color_cycle = False
        _cycle_color()

    tick_delta = time.time() - last_time
    last_time = time.time()

    if flip:
        frame = cv2.flip(frame, 1)

    if _frame_counter % 2 == 0:
        detect_frame = cv2.resize(frame, (640, 360))
        rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= _last_video_ts_ms:
            ts_ms = _last_video_ts_ms + 1
        _last_video_ts_ms = ts_ms
        _last_results = _hand_detector.detect_for_video(mp_image, ts_ms).hand_landmarks
    results = _last_results
    _frame_counter += 1

    buttons_pressed = set()

    if results:
        for hand_landmarks in results:
            # Display hand landmarks


            # Get finger positions
            index_finger_tip = hand_landmarks[INDEX_FINGER_TIP]
            thumb_finger_tip = hand_landmarks[THUMB_TIP]
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            thumb_x, thumb_y = int(thumb_finger_tip.x * frame.shape[1]), int(thumb_finger_tip.y * frame.shape[0])

            # No gesture-based tool switching - tool will be set manually from outside

            # Check for pinch gesture (index and thumb close)
            is_pinching = index_thumb_close(hand_landmarks)

            # UI strip is horizontally mirrored when flip=False so the
            # downstream self-view mirror restores readability; hit-test
            # has to mirror the hand x to match where we drew the button.
            if flip:
                hand_point_ui = (index_x, index_y)
            else:
                hand_point_ui = (frame.shape[1] - 1 - index_x, index_y)
            over_button = False
            for btn in _buttons:
                if btn.contains(hand_point_ui):
                    over_button = True
                    if is_pinching:
                        buttons_pressed.add(id(btn))
                    break
            if over_button and is_pinching:
                continue

            # Process based on current tool
            if tool == 'draw':
                # Drawing mode logic
                if is_pinching:
                    h, w, _ = frame.shape
                    x, y = index_x, index_y

                    if not thickness:
                        thickness = 5

                    if drawing:
                        if last_x is not None and last_y is not None:
                            # Exponential smoothing against the previous
                            # smoothed point — mediapipe landmarks jitter a
                            # few pixels per frame, which reads as a shaky
                            # stroke if fed straight to cv2.line.
                            a = STROKE_SMOOTH_ALPHA
                            sx = int(a * x + (1 - a) * last_x)
                            sy = int(a * y + (1 - a) * last_y)
                            cv2.line(canvas, (last_x, last_y), (sx, sy),
                                     get_draw_color(), thickness=thickness)
                            canvas_dirty = True
                            last_x, last_y = sx, sy
                        else:
                            last_x, last_y = x, y
                    else:
                        last_x, last_y = None, None

                    drawing = True
                else:
                    draw_cooldown_threshold += tick_delta
                    if draw_cooldown_threshold > 1:
                        draw_cooldown_threshold = 0
                        drawing = False

            elif tool == 'select':
                # Selection mode logic
                hand_position = (index_x, index_y)
                
                # Draw a cursor at the index finger tip
                cv2.circle(frame, hand_position, 10, (0, 255, 255), -1)
                
                if is_pinching:
                    # If we weren't already moving a shape, try to select one
                    if not is_moving_shape:
                        if select_shape(hand_position):
                            is_moving_shape = True
                            move_start_pos = hand_position
                            print(f"Selected shape {selected_shape_idx}")
                    # If we already have a shape selected, move it
                    elif selected_shape_idx is not None and move_start_pos is not None:
                        # Calculate movement offset
                        dx = hand_position[0] - move_start_pos[0]
                        dy = hand_position[1] - move_start_pos[1]
                        
                        # Apply movement if significant
                        if abs(dx) > 5 or abs(dy) > 5:
                            move_shape(selected_shape_idx, (dx, dy))
                            move_start_pos = hand_position
                else:
                    # Release the shape when pinch gesture ends
                    is_moving_shape = False
                    move_start_pos = None

    else:
        draw_cooldown_threshold += tick_delta
        if draw_cooldown_threshold > 1:
            draw_cooldown_threshold = 0
            drawing = False

    for btn in _buttons:
        btn.tick(tick_delta, pressed=id(btn) in buttons_pressed)

    # Process drawing completion
    if tool == 'draw' and not drawing:
        last_x, last_y = None, None
        if canvas_dirty:

            pred_probab, pred_class = keras_predict(canvas)

            pred_shape = ['square', 'circle', 'triangle', 'line', 'arrow'][int(pred_class)]

            if pred_probab >= 0.8:
                correct_image(canvas, pred_shape)
            else:
                permanent_canvas = cv2.addWeighted(permanent_canvas, 1, canvas, 1, 0)
                _push_undo()
            canvas[:] = 0
            canvas_dirty = False
            thickness = None

    # Highlight selected shape
    if tool == 'select' and selected_shape_idx is not None and 0 <= selected_shape_idx < len(shapes):
        selected_shape = shapes[selected_shape_idx]
        if 'bbox' in selected_shape:
            x1, y1, x2, y2 = selected_shape['bbox']
            # Draw a semi-transparent yellow rectangle around the selected shape
            highlight = frame.copy()
            cv2.rectangle(highlight, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 255), 2)
            cv2.addWeighted(highlight, 0.4, frame, 0.6, 0, frame)

    # Overlay strokes onto the frame at STROKE_OPACITY. Uses a mask-based
    # replace/blend instead of additive cv2.addWeighted so colours stay
    # true (red strokes look red, not pink). canvas is only scanned when
    # a stroke is in progress.
    _overlay_strokes(frame, permanent_canvas, STROKE_OPACITY)
    if canvas_dirty:
        _overlay_strokes(frame, canvas, STROKE_OPACITY)

    # Render UI to a narrow BGRA strip (alpha preserved until composite
    # so translucent pills blend against the live video). Flip the strip
    # when flip=False so text/icons read correctly through a downstream
    # self-view mirror. 130 rows keeps the cost negligible.
    ui_h = min(130, frame.shape[0])
    ui_layer = np.zeros((ui_h, frame.shape[1], 4), dtype=np.uint8)
    for btn in _buttons:
        btn.draw(ui_layer)
    if not flip:
        ui_layer = cv2.flip(ui_layer, 1)
    _composite_bgra(frame[:ui_h], ui_layer)

    return frame


def set_tool(new_tool):
    """Set the current tool to either 'draw' or 'select'"""
    global tool
    if new_tool in ['draw', 'select']:
        tool = new_tool
        print(f"Tool set to: {tool}")
    else:
        print(f"Invalid tool: {new_tool}. Use 'draw' or 'select'.")