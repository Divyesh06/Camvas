import os
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import gc
import numpy as np
from softcam_python import softcam
import camera_worker

FRAME_W = 1920
FRAME_H = 1080
SLOT_BYTES = FRAME_W * FRAME_H * 3
SHM_BYTES = SLOT_BYTES * 2
GRACE_SECONDS = 20

stop_signal = True
cam = None

worker = None
shm = None
slots = None
latest_idx = None
frame_available = None
capture_active = None
shutdown_event = None
pending_out = None

placeholder_frame = None


def is_frozen():
    return getattr(sys, 'frozen', False)


def stop_camera():
    global stop_signal
    stop_signal = True


def _placeholder_path():
    if is_frozen():
        return os.path.join(os.path.dirname(sys.executable), 'Camvas.bmp')
    return os.path.join('assets', 'Camvas.bmp')


def _load_placeholder():
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage

    img = QImage(_placeholder_path())
    if img.isNull():
        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    img = img.convertToFormat(QImage.Format_RGB888)
    img = img.scaled(FRAME_W, FRAME_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    ptr = img.bits()
    ptr.setsize(bpl * h)
    raw = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, bpl))
    rgb = raw[:, :w * 3].reshape((h, w, 3))
    bgr = rgb[..., ::-1]

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    y0 = (FRAME_H - h) // 2
    x0 = (FRAME_W - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = bgr
    return canvas


def _spawn_worker():
    global worker, shm, slots, latest_idx, frame_available, capture_active, shutdown_event, pending_out

    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    slots = [
        np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf, offset=0),
        np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf, offset=SLOT_BYTES),
    ]
    latest_idx = mp.Value('i', 0)
    frame_available = mp.Event()
    capture_active = mp.Event()
    shutdown_event = mp.Event()
    if pending_out is None:
        pending_out = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    worker = mp.Process(
        target=camera_worker.run,
        args=(shm.name, latest_idx, frame_available, capture_active, shutdown_event),
        daemon=True,
    )
    worker.start()


def _kill_worker():
    global worker, shm, slots, latest_idx, frame_available, capture_active, shutdown_event

    if shutdown_event is not None:
        shutdown_event.set()
    if capture_active is not None:
        capture_active.set()  # unblock worker if it's idling

    if worker is not None:
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)
            if worker.is_alive():
                worker.kill()
                worker.join()

    if shm is not None:
        slots = None
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    worker = None
    shm = None
    slots = None
    latest_idx = None
    frame_available = None
    capture_active = None
    shutdown_event = None


def _respawn_worker():
    _kill_worker()
    _spawn_worker()


def main(started_callback, disconnected_callback=None):
    global stop_signal, cam, placeholder_frame

    if cam:
        return

    stop_signal = False
    cam = softcam.camera(FRAME_W, FRAME_H, 60)

    while not stop_signal:
        cam.wait_for_connection(timeout=300)
        time.sleep(1)
        if cam.is_connected():
            break

    if stop_signal:
        cam.delete()
        cam = None
        stop_signal = False
        return

    if placeholder_frame is None:
        placeholder_frame = _load_placeholder()

    _spawn_worker()
    capture_active.set()
    started_callback()

    seen_real_frame = False
    grace_deadline = None
    in_grace = False
    was_stopped = False
    frame_count = 0
    fps_start = time.time()

    try:
        while not stop_signal:
            if worker is not None and not worker.is_alive():
                _respawn_worker()
                if not in_grace:
                    capture_active.set()

            now = time.time()
            if cam.is_connected():
                if in_grace:
                    time.sleep(1)
                    if not cam.is_connected():
                        continue
                    in_grace = False
                    grace_deadline = None
                    if capture_active is not None:
                        capture_active.set()
                    if frame_available is not None:
                        frame_available.clear()
                    started_callback()

                if frame_available.wait(timeout=0.05):
                    frame_available.clear()
                    np.copyto(pending_out, slots[latest_idx.value])
                    cam.send_frame(pending_out)
                    seen_real_frame = True
                else:
                    if seen_real_frame:
                        cam.send_frame(pending_out)
                    else:
                        cam.send_frame(placeholder_frame)

                frame_count += 1
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    print(f"FPS: {frame_count / elapsed:.1f}")
                    frame_count = 0
                    fps_start = now
            else:
                if not in_grace:
                    in_grace = True
                    grace_deadline = now + GRACE_SECONDS
                    if capture_active is not None:
                        capture_active.clear()
                    if disconnected_callback:
                        disconnected_callback()

                if now >= grace_deadline:
                    break

                cam.wait_for_connection(timeout=0.5)

        was_stopped = stop_signal
    finally:
        _kill_worker()
        cam.delete()
        
        gc.collect()
        cam = None
        stop_signal = False

    if disconnected_callback and not was_stopped:
        disconnected_callback()


if __name__ == "__main__":
    main(lambda: None)
