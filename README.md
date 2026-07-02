# Finger Detection & Counting (Webcam) — Python + OpenCV + MediaPipe

Real-time hand tracking that counts how many fingers are raised, using your
laptop/USB webcam. Built to run locally in VSCode.

## How it works

1. **MediaPipe Hands** detects 21 landmarks per hand from each webcam frame.
2. For each fingertip landmark, the script compares its position to a lower
   joint on the same finger:
   - **Thumb** — compared on the X-axis (thumb moves sideways).
   - **Index / Middle / Ring / Pinky** — compared on the Y-axis (tip above
     the joint = finger extended).
3. The count (0–5) and skeleton overlay are drawn on the live video feed.

## Project structure

```
finger-detection/
├── finger_detection.py   # main script
├── requirements.txt      # dependencies
└── README.md
```

## Setup (VSCode, Windows/macOS/Linux)

1. **Install Python 3.9–3.11** (MediaPipe doesn't yet support every latest
   Python version — 3.10 is the safest bet). Check with:
   ```bash
   python --version
   ```

2. **Open the folder in VSCode**, then open a terminal (`` Ctrl+` ``).

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Select the interpreter in VSCode:** `Ctrl+Shift+P` → "Python: Select
   Interpreter" → choose the `venv` one.

6. **Run it:**
   ```bash
   python finger_detection.py
   ```
   A window will open showing your webcam feed with hand landmarks and a
   live finger count. Press **q** to quit.

## Troubleshooting

- **Webcam doesn't open / black window:** try changing
  `cv2.VideoCapture(0)` to `1` or `2` in `main()` — some laptops list an
  internal camera at a different index than an external USB webcam.
- **`ImportError: DLL load failed` (Windows):** update to the latest
  Microsoft Visual C++ Redistributable, then reinstall `mediapipe`.
- **Low FPS:** lower the capture resolution (edit the
  `cap.set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT, ...)` lines) or run on better
  lighting — MediaPipe tracks faster with clear hand contrast.
- **MediaPipe install fails on Python 3.12+:** switch to Python 3.10 or
  3.11 in your virtual environment.

## Extending this project

Some natural next steps if you want to build on it:
- Detect **both hands** (`max_hands=2`) and sum total fingers across both.
- Map finger counts to actions — e.g. 5 fingers = take a screenshot, 1
  finger = draw on a virtual canvas (AR-style painting apps use this).
- Swap counting logic for **gesture recognition** (thumbs up, peace sign,
  OK sign) using landmark angle/distance patterns instead of simple up/down.
- Log finger counts with timestamps to a CSV for a simple sign-language
  digit trainer.
