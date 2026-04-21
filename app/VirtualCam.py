import importlib
from softcam_python import softcam
import sys

stop_signal = True
cam = None

cv2 = None
frame_processor = None


def stop_camera():
    global stop_signal
    stop_signal = True

def load_modules():
    global cv2, frame_processor
    cv2 = importlib.import_module('cv2')
    frame_processor = importlib.import_module('frame_processor')

def main(started_callback, disconnected_callback=None):
    global stop_signal, cam, cv2, frame_processor

    if cam:
        return

    stop_signal = False
    cam = softcam.camera(1920, 1080, 60)

    import time
    while not stop_signal:
        cam.wait_for_connection(timeout=300)
        time.sleep(1)  # softcam reports connected too early; let is_connected settle
        if cam.is_connected():
            break

    if stop_signal:
        cam.delete()
        cam = None
        stop_signal = False
        return

    if not cv2 or not frame_processor:
        load_modules()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = 1920
    frame_height = 1080
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    first_time = True
    needs_resize = False

    started_callback()

    frame_count = 0
    fps_start = time.time()

    while cam.is_connected() and not stop_signal:

        ret, frame = cap.read()

        if first_time:
            frame_processor.init_state(frame.shape)
            needs_resize = frame.shape != (frame_height, frame_width, 3)
            first_time = False

        frame = frame_processor.process_frame(frame, False)

        if needs_resize:
            frame = cv2.resize(frame, (frame_width, frame_height), frame)

        cam.send_frame(frame)

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            print(f"FPS: {frame_count / elapsed:.1f}")
            frame_count = 0
            fps_start = time.time()

    was_stopped = stop_signal
    cap.release()
    cam.delete()
    cam = None

    stop_signal = False

    for mod in ['cv2', 'frame_processor']:
        if mod in sys.modules:
            del sys.modules[mod]

    if disconnected_callback and not was_stopped:
        disconnected_callback()



if __name__ == "__main__":
    main(lambda: None)
