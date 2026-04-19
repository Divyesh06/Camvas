
# We'll keep this function for external use but it won't be triggered by gestures
def set_tool(new_tool):
    """Set the current tool to either 'draw' or 'select'"""
    global tool
    if new_tool in ['draw', 'select']:
        tool = new_tool
        print(f"Tool set to: {tool}")
    else:
        print(f"Invalid tool: {new_tool}. Use 'draw' or 'select'.")
        
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


import time
import sys
import onnxruntime as ort
import os
import math
import threading
#import keyboard  # You'll need to install this: pip install keyboard
def is_frozen():
    return getattr(sys, 'frozen', False)

if not is_frozen():
    model_path = "model/ShapeDetection.onnx"
    hand_model_path = "model/hand_landmarker.task"
else:
    app_dir = os.path.dirname(sys.executable)
    model_path = os.path.join(app_dir, 'model', 'ShapeDetection.onnx')
    hand_model_path = os.path.join(app_dir, 'model', 'hand_landmarker.task')

onnx_session = ort.InferenceSession(model_path)
input_name = onnx_session.get_inputs()[0].name

THUMB_TIP = 4
INDEX_FINGER_TIP = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

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


def draw_hand_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)

# Global variables
canvas = None
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


def init_state(frame_shape):
    """Initializes or resets all global variables to a clean state."""
    global canvas, permanent_canvas, undo, redo
    global init, drawing, last_x, last_y
    global draw_cooldown_threshold, undo_cooldown_threshold
    global tool, shapes, selected_shape_idx, is_moving_shape, move_start_pos

    canvas = np.zeros(frame_shape, dtype=np.uint8)
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
    
    # Set up keyboard event listener
  #  start_keyboard_listener()

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
    
    shape_classes = ['square', 'circle', 'triangle', 'line', 'unknown']
    confidence = float(np.max(pred_probab))

    print(f"I am {confidence * 100:.2f}% sure that this is a {shape_classes[pred_class]}")
    return confidence, pred_class

def correct_image(image, shape):
    global permanent_canvas, shapes
    leftmost, topmost, rightmost, bottommost, center = find_extreme_non_zero_points(image)

    # Create a new shape and store its properties
    new_shape = {
        'type': shape,
        'points': [],
        'center': center,
        'bbox': (leftmost[0], topmost[1], rightmost[0], bottommost[1])  # x1, y1, x2, y2
    }
    
    if shape == 'circle':
        radius = center[0] - leftmost[0]
        cv2.circle(permanent_canvas, center, radius, (255, 0, 0), 5)
        new_shape['points'] = [center, radius]
    elif shape == 'square':
        cv2.rectangle(permanent_canvas, (leftmost[0], topmost[1]), (rightmost[0], bottommost[1]), (255, 0, 0), 5)
        new_shape['points'] = [(leftmost[0], topmost[1]), (rightmost[0], bottommost[1])]
    elif shape == 'triangle':
        if abs(bottommost[1] - min(leftmost[1], rightmost[1])) > abs(topmost[1] - min(leftmost[1], rightmost[1])):
            topmost = bottommost
        cv2.line(permanent_canvas, leftmost, topmost, (255, 0, 0), 5)
        cv2.line(permanent_canvas, rightmost, topmost, (255, 0, 0), 5)
        cv2.line(permanent_canvas, leftmost, rightmost, (255, 0, 0), 5)
        new_shape['points'] = [leftmost, topmost, rightmost]
    elif shape == 'line':
        cv2.line(permanent_canvas, leftmost, rightmost, (255, 0, 0), 5)
        new_shape['points'] = [leftmost, rightmost]

    # Add shape to our list of shapes
    shapes.append(new_shape)
    
    undo.append(permanent_canvas.copy())
    redo.clear()

def find_extreme_non_zero_points(img):
    non_zero_coords = np.argwhere(img != 0)
    y_coords = non_zero_coords[:, 0]
    x_coords = non_zero_coords[:, 1]

    leftmost = (x_coords.min(), y_coords[x_coords.argmin()])
    rightmost = (x_coords.max(), y_coords[x_coords.argmax()])
    topmost = (x_coords[y_coords.argmin()], y_coords.min())
    bottommost = (x_coords[y_coords.argmax()], y_coords.max())
    center = ((leftmost[0] + rightmost[0]) // 2, (leftmost[1] + rightmost[1]) // 2)

    return leftmost, topmost, rightmost, bottommost, center

def keras_process_image(img):
    img = auto_crop(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 28, 28, 1))
    return img

def auto_crop(img):
    non_zero_rows = np.any(img != 0, axis=(1, 2))
    non_zero_cols = np.any(img != 0, axis=(0, 2))
    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]
    return img[row_start:row_end + 1, col_start:col_end + 1]

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
        
    elif shape_type == 'line':
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
        
    elif shape['type'] == 'line':
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
    
    undo.append(permanent_canvas.copy())

def redraw_all_shapes():
    """Redraw all shapes on the permanent canvas"""
    global permanent_canvas, shapes
    
    # Clear the canvas
    permanent_canvas = np.zeros_like(permanent_canvas)
    
    # Redraw each shape
    for shape in shapes:
        if shape['type'] == 'circle':
            center, radius = shape['points']
            cv2.circle(permanent_canvas, center, radius, (255, 0, 0), 5)
            
        elif shape['type'] == 'square':
            (x1, y1), (x2, y2) = shape['points']
            cv2.rectangle(permanent_canvas, (x1, y1), (x2, y2), (255, 0, 0), 5)
            
        elif shape['type'] == 'triangle':
            p1, p2, p3 = shape['points']
            cv2.line(permanent_canvas, p1, p2, (255, 0, 0), 5)
            cv2.line(permanent_canvas, p2, p3, (255, 0, 0), 5)
            cv2.line(permanent_canvas, p3, p1, (255, 0, 0), 5)
            
        elif shape['type'] == 'line':
            p1, p2 = shape['points']
            cv2.line(permanent_canvas, p1, p2, (255, 0, 0), 5)

# Set up keyboard event for tool switching
def setup_keyboard_events():
    """Set up keyboard event listener for tool switching"""
    keyboard.add_hotkey('s', lambda: set_tool('select'))
    keyboard.add_hotkey('d', lambda: set_tool('draw'))
    print("Keyboard shortcuts enabled: Press 's' for Select mode, 'd' for Draw mode")

# Start keyboard listener in a separate thread
def start_keyboard_listener():
    """Start keyboard listener in a separate thread"""
    try:
        # Create and start keyboard listener thread
        keyboard_thread = threading.Thread(target=keyboard.wait)
        keyboard_thread.daemon = True  # Thread will close when main program exits
        keyboard_thread.start()
        setup_keyboard_events()
    except Exception as e:
        print(f"Error setting up keyboard listener: {e}")
        print("Tool switching via keyboard will not be available.")

# Call this function at the beginning of your main loop/program
# start_keyboard_listener()

def process_frame(frame, flip=False):
    global canvas, last_time, draw_cooldown_threshold, thickness
    global init, drawing, permanent_canvas, last_x, last_y
    global tool, selected_shape_idx, is_moving_shape, move_start_pos
    global _last_video_ts_ms

    if not init:
        init_state(frame.shape)

    tick_delta = time.time() - last_time
    last_time = time.time()

    if flip:
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    ts_ms = int(time.monotonic() * 1000)
    if ts_ms <= _last_video_ts_ms:
        ts_ms = _last_video_ts_ms + 1
    _last_video_ts_ms = ts_ms
    results = _hand_detector.detect_for_video(mp_image, ts_ms)

    # Add key instructions to the frame
    cv2.putText(frame, f"Tool: {tool} (Press 'd' for Draw, 's' for Select)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Display hand landmarks
            draw_hand_landmarks(frame, hand_landmarks)

            # Get finger positions
            index_finger_tip = hand_landmarks[INDEX_FINGER_TIP]
            thumb_finger_tip = hand_landmarks[THUMB_TIP]
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            thumb_x, thumb_y = int(thumb_finger_tip.x * frame.shape[1]), int(thumb_finger_tip.y * frame.shape[0])

            # No gesture-based tool switching - tool will be set manually from outside

            # Check for pinch gesture (index and thumb close)
            is_pinching = index_thumb_close(hand_landmarks)
            
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
                            cv2.line(canvas, (last_x, last_y), (x, y), (255, 0, 0), thickness=thickness)
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

    # Process drawing completion
    if tool == 'draw' and not drawing:
        last_x, last_y = None, None
        if np.count_nonzero(canvas):

            pred_probab, pred_class = keras_predict(canvas)

            pred_shape = ['square', 'circle', 'triangle', 'line', 'unknown'][int(pred_class)]
            
            if pred_probab >= 0.99 and pred_shape != 'unknown':
                correct_image(canvas, pred_shape)
            else:
                permanent_canvas = cv2.addWeighted(permanent_canvas, 1, canvas, 1, 0)
                undo.append(permanent_canvas.copy())
            canvas = np.zeros_like(frame)
            thickness = None

    # Highlight selected shape
    if tool == 'select' and selected_shape_idx is not None:
        selected_shape = shapes[selected_shape_idx]
        if 'bbox' in selected_shape:
            x1, y1, x2, y2 = selected_shape['bbox']
            # Draw a semi-transparent yellow rectangle around the selected shape
            highlight = frame.copy()
            cv2.rectangle(highlight, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 255), 2)
            cv2.addWeighted(highlight, 0.4, frame, 0.6, 0, frame)

    # Combine drawing layers
    frame = cv2.addWeighted(frame, 0.5, canvas, 1, 0)
    frame = cv2.addWeighted(frame, 1, permanent_canvas, 1, 0)

    return frame