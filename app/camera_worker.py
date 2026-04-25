import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FRAME_W = 1920
FRAME_H = 1080
SLOT_BYTES = FRAME_W * FRAME_H * 3


def run(shm_name, latest_idx, frame_available, capture_active, shutdown):
    import numpy as np
    from multiprocessing import shared_memory
    import cv2
    import frame_processor

    shm = shared_memory.SharedMemory(name=shm_name)
    slots = [
        np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf, offset=0),
        np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf, offset=SLOT_BYTES),
    ]

    cap = None
    first = True
    needs_resize = False

    def open_cap():
        c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        c.set(cv2.CAP_PROP_FPS, 30)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        return c

    try:
        while not shutdown.is_set():
            if not capture_active.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                    first = True
                if shutdown.wait(timeout=0.1):
                    break
                continue

            if cap is None:
                cap = open_cap()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if first:
                frame_processor.init_state(frame.shape)
                needs_resize = frame.shape != (FRAME_H, FRAME_W, 3)
                first = False

            frame = frame_processor.process_frame(frame, False)

            if needs_resize:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            next_idx = 1 - latest_idx.value
            np.copyto(slots[next_idx], frame)
            latest_idx.value = next_idx
            frame_available.set()
    finally:
        if cap is not None:
            cap.release()
        shm.close()
