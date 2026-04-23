# Camvas

Camvas turns your webcam feed into a hand-drawable canvas. Draw directly on your video feed with your hand and let the AI correct your drawing into a clean geometric shape.

The annotated feed is exposed as a virtual camera device, so any app (Zoom, Meet, Teams, Discord, Slack huddles, browser tabs) can use it as a webcam source. 

The virtual camera is currently supported only on Windows (using [Softcam](https://github.com/tshino/softcam)
)

## How it works

Every webcam frame flows through a small pipeline of layers:

1. **Webcam Layer** — Pulls the feed directly from your webcam.
2. **Hand tracking** — a MediaPipe hand-landmark model locates your index-tip and thumb-tip in the frame.
3. **Gesture interpretation** — if the two tips are close enough, that counts as a pinch. The pinch either hits an on-screen button (if the finger is over one) or starts / continues a drawing stroke.
4. **Classification** — when the pinch releases and the stroke ends, the stroke canvas is cropped, binarized, resized to 28×28, and fed through a small CNN. The model returns a shape label and a confidence score.
5. **Correction** — if the model is confident enough, the raw stroke is replaced with a clean geometric version (works by analysing vertices and edges). Uncertain strokes are kept as raw ink instead.
6. **Compositing** — the output frame is built in layers: webcam image at the base, then finalized drawings, then any in-progress stroke, then the UI strip (buttons and label) at the top.
7. **Virtual-camera output** — the composed frame is pushed to a virtual webcam device, which your video or streaming app reads as if it were a real camera.

## Download

Prebuilt installer (Windows):

[**Download the latest release**](https://github.com/Divyesh06/Camvas/releases/latest)

## Features

- **Works anywhere** - Just switch the virtual camera in any app
- **Auto-shape-correction** for squares, circles, triangles, lines, and arrows.
- **Toolbar** - Displays a toolbar directly over your video feed with buttons that you literally "hold" to activate. Allows moving drawings, changing color, undo, and clear. Also supports shortcuts
- **Real-time performance** - Optimised to run smoothly in real-time at high frame rate and minimal CPU usage.
- **Runs in background** - Can run on startup and stay ready in background. Automatically clears memory when not in use.

## Running locally

**Prerequisites**

Download uv from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)


```bash
git clone https://github.com/Divyesh06/Camvas.git
cd Camvas
uv venv
uv pip install -r requirements.txt
uv run poe dev
```
