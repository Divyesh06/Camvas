import sys
import os
import importlib
from time import sleep

stop_signal = True
reset_signal = False
cam = None

cv2 = None
frame_processor = None

from softcam_python import softcam

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 60


def stop_camera():
    global stop_signal
    stop_signal = True


def reset_camera():
    global reset_signal
    reset_signal = True


def load_modules():
    global cv2, frame_processor
    cv2 = importlib.import_module('cv2')
    frame_processor = importlib.import_module('frame_processor')


def _setup_camera():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    vc.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    vc.set(cv2.CAP_PROP_FPS, FPS)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not vc.isOpened():
        vc = cv2.VideoCapture(0)
        if not vc.isOpened():
            raise RuntimeError("Could not open video source")

    return vc


def main(started_callback, disconnected_callback=None):
    global stop_signal
    global reset_signal
    global cam

    if cam:
        return

    stop_signal = False
    cam = softcam.camera(FRAME_WIDTH, FRAME_HEIGHT, FPS)

    if not cv2 and not frame_processor:
        load_modules()

    vc = None
    first_time = True

    while not stop_signal:
        while not stop_signal:
            print("off")
            cam.wait_for_connection(timeout=300)
            sleep(3)  # softcam reports connected too early; this delay lets the pipe settle
            if cam.is_connected():
                break

        if stop_signal:
            break

        reset_signal = False
        started_callback()

        while not stop_signal:
            if reset_signal:
                reset_signal = False
                if vc:
                    vc.release()
                    vc = None
                break

            if vc and vc.isOpened():
                ret, frame = vc.read()

                if first_time:
                    frame_processor.init_state(frame.shape)
                    first_time = False

                frame = frame_processor.process_frame(frame, False)

                if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), frame)

                cam.send_frame(frame)
            else:
                print("setup")
                vc = _setup_camera()

            if not cam.is_connected():
                if vc:
                    vc.release()  # must release before the next _setup_camera() or DShow deadlocks
                    vc = None
                if disconnected_callback:
                    disconnected_callback()
                break

    print("Camera stopped")
    if vc:
        vc.release()
    cam.delete()
    cam = None
    stop_signal = False


if __name__ == "__main__":
    main(lambda: None)
