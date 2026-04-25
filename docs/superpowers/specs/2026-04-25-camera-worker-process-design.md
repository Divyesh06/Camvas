# Camera Worker Process — Design

**Date:** 2026-04-25
**Status:** Approved (pending user review of this doc)

## Problem

`cv2`, `mediapipe`, and `onnxruntime` together hold hundreds of MB of native
state. Camvas spends most of its lifetime idle in `softcam.wait_for_connection`
waiting for a consumer (Zoom, Meet, etc.) to open the virtual camera. While
idle, that memory should not be resident.

The current code in `app/VirtualCam.py` tries to free it by lazy-importing
`cv2` and `frame_processor`, then doing `del sys.modules[...]` after the
consumer disconnects. This does not work — C extensions retain native
allocations regardless of `sys.modules` state.

## Goal

Keep the heavy modules out of the main process entirely. Load them in a
short-lived worker process that can be killed (and its memory truly released)
when no consumer is connected.

## Architecture

Two processes:

- **Main process** — tray UI, `softcam`, watchdog. Imports stay light: no
  `cv2`, no `mediapipe`, no `onnxruntime`. Owns the virtual camera
  registration (which must persist for the OS to advertise the device).
- **Worker process** (`multiprocessing.Process`) — imports `cv2` +
  `frame_processor`, opens `cv2.VideoCapture(0, CAP_DSHOW)`, runs the
  `process_frame` loop, writes processed BGR frames into shared memory.
  Spawned on demand, killed when not needed.

`softcam` stays in main because `softcam.camera.is_connected()` is the
watchdog signal that tells main when to spin the worker up and tear it down.

## IPC

One shared-memory frame buffer plus two events:

- `multiprocessing.shared_memory.SharedMemory` sized for one 1920×1080×3 BGR
  frame (~6 MB). Single buffer, not double — at 30 fps the memcpy is
  negligible and a single buffer is simpler.
- `multiprocessing.Event` `frame_ready` — worker sets after writing a frame;
  main clears after copying.
- `multiprocessing.Event` `shutdown` — main sets to ask the worker to exit
  cleanly.
- `multiprocessing.Pipe` (worker → main) for status messages: `"ready"` (sent
  after first successful `cap.read()` so main can stop the placeholder),
  `"error: <msg>"` for surfacing failures.

No per-frame pickling. Worker `numpy.ndarray.tofile`-style writes via a
numpy view backed by `SharedMemory.buf`.

## Lifecycle

1. **App start.** Main creates `softcam.camera(1920, 1080, 60)` and calls
   `wait_for_connection(timeout=300)` in a loop. Worker is **not** spawned.
2. **Consumer connects.** Main spawns the worker, allocates the shared
   memory, and immediately starts sending the **placeholder frame**
   (`assets/Camvas.bmp` resized to 1920×1080) via `cam.send_frame` on each
   `is_connected()` tick.
3. **Worker ready.** Worker finishes imports, opens `VideoCapture`, gets its
   first frame, calls `frame_processor.init_state(frame.shape)`, writes the
   first processed frame to shared memory, sends `"ready"` on the pipe, sets
   `frame_ready`. Main switches from placeholder to reading from shared
   memory: `wait(frame_ready)` → copy → clear → `cam.send_frame`.
4. **Consumer disconnects.** Main starts a **20-second grace timer** but
   keeps the worker alive. If a consumer reconnects within 20 s, cancel the
   timer and resume reading from shared memory (no cold-start cost). If the
   timer fires, main sets `shutdown`, `join(timeout=2)`, then `terminate()`
   if still alive, and unlinks shared memory.
5. **User clicks Stop in the tray.** Main sets `shutdown` immediately, kills
   the worker, deletes the softcam object.
6. **Worker crashes mid-session.** Detected via `process.is_alive() == False`
   while the consumer is still connected. Main respawns the worker; reverts
   to the placeholder until the new worker sends `"ready"`. No backoff, no
   crash-loop guard — kept simple per project scope.

## File layout

- **New:** `app/camera_worker.py` — worker entry point. Owns `cv2` and
  `frame_processor` imports. Exposes a single `run(shm_name, frame_ready,
  shutdown, status_pipe)` function used as the `multiprocessing.Process`
  target.
- **Modified:** `app/VirtualCam.py` — gutted of cv2/frame_processor. Owns
  softcam, the watchdog loop, the worker spawn/kill logic, the placeholder
  loader, and the 20s grace timer.
- **Modified:** `app/main.py` — no functional change expected; `VirtualCam`
  still exposes `main(started_callback, disconnected_callback)` and
  `stop_camera()`.
- **Reused:** `assets/Camvas.bmp` as the placeholder source. Loaded once in
  main via `PyQt5.QtGui.QImage` (PyQt5 is already imported by `status_app`),
  resized to 1920×1080, and converted to a contiguous BGR `numpy.ndarray`.
  No `cv2` in main.

## What gets removed

- `del sys.modules['cv2']` / `del sys.modules['frame_processor']` block in
  `VirtualCam.py`.
- `importlib.import_module('cv2')` lazy-load dance and the `cv2 = None` /
  `frame_processor = None` module globals.

## Out of scope

- Persisting drawing state (canvas, undo/redo, shapes) across worker
  restarts. Today these reset on every new session via `init_state()`;
  behavior is unchanged.
- Backoff or restart-limit on worker crashes.
- Pre-warming the worker before the first connection. The first connect
  after launch will pay the 1–3 s cold-start cost and show the placeholder
  during that window.
- Double-buffering the shared-memory frame.

## Open questions

None at design time. Implementation may surface details around:

- Exact behavior of `cam.send_frame` if called at less than 30 fps during
  the placeholder window — should be fine (consumer just sees the static
  image).
