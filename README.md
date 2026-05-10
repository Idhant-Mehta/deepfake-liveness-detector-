# Deepfake Liveness Detector

A real-time computer vision application that detects facial liveness to prevent presentation attacks (like static images or pre-recorded deepfake videos). The program applies facial landmark detection to monitor eye blinks, verifying the presence of a live user.

## Features
- **Real-Time Blink Detection:** Monitors Eye Aspect Ratio (EAR) using precise facial landmarks.
- **Liveness Status UI:** Overlays live feedback directly onto the video feed (e.g., "Live (Blinks Detected)" vs "Spoof/Fake").
- **Deepfake Mitigation:** Adds a simple but effective layer of security against static spoofing and primitive deepfakes.

## Prerequisites
Ensure you have Python 3.7+ installed. The project relies on the following libraries:
- OpenCV (`cv2`) for video capture and image rendering.
- MediaPipe (`mediapipe`) for high-fidelity facial landmark tracking (`refine_landmarks=True`).
- NumPy (`numpy`) for mathematical operations.

## Installation

1. Clone or download this repository.
2. Install the required Python dependencies:

```bash
pip install opencv-python mediapipe numpy

