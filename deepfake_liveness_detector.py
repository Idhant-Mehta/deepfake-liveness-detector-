import cv2
import mediapipe as mp
import math
import time
import argparse
from collections import deque

def euclidean(p1, p2):
    return math.dist(p1, p2)

def landmark_xy(landmark, w, h):
    return (landmark.x * w, landmark.y * h)

def eye_aspect_ratio(landmarks, w, h, top_ids, bottom_ids, corner_left_id, corner_right_id):
    top_points = [landmark_xy(landmarks[i], w, h) for i in top_ids]
    bottom_points = [landmark_xy(landmarks[i], w, h) for i in bottom_ids]
    left_corner = landmark_xy(landmarks[corner_left_id], w, h)
    right_corner = landmark_xy(landmarks[corner_right_id], w, h)

    verts = [abs(tp[1] - bp[1]) for tp, bp in zip(top_points, bottom_points)]
    avg_vert = sum(verts) / len(verts)

    width = euclidean(left_corner, right_corner) + 1e-6
    return avg_vert / width

def detect_liveness(video_source=0):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    delay = int(1000 / fps) if isinstance(video_source, str) else 1

    start_time = time.time()
    eye_closed = False
    blink_timestamps = deque()
    ear_buffer = deque(maxlen=3)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            liveness_status = "Status: Checking Liveness..."
            color = (0, 255, 255)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    lm = landmarks.landmark

                    # Left Eye EAR
                    left_top = [160, 158]
                    left_bottom = [144, 153]
                    left_ear = eye_aspect_ratio(lm, w, h, left_top, left_bottom, 33, 133)

                    # Right Eye EAR
                    right_top = [385, 387]
                    right_bottom = [380, 373]
                    right_ear = eye_aspect_ratio(lm, w, h, right_top, right_bottom, 362, 263)

                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_buffer.append(avg_ear)
                    smoothed_ear = sum(ear_buffer) / len(ear_buffer)

                    if smoothed_ear < 0.20:
                        eye_closed = True
                    elif smoothed_ear > 0.22 and eye_closed:
                        eye_closed = False
                        blink_timestamps.append(time.time())

                current_time = time.time()
                # Remove blinks older than 5 seconds
                while blink_timestamps and current_time - blink_timestamps[0] > 5.0:
                    blink_timestamps.popleft()

                if current_time - start_time < 3.0 and len(blink_timestamps) == 0:
                    liveness_status = "Status: Checking Liveness..."
                    color = (0, 255, 255)
                elif len(blink_timestamps) > 0:
                    liveness_status = "Status: Live (Blinks Detected)"
                    color = (0, 255, 0)
                else:
                    liveness_status = "Status: Fake/Spoof Warning!"
                    color = (0, 0, 255)

            cv2.putText(frame, liveness_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "Press 'q' to quit", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Liveness Detector', frame)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Liveness Detector")
    parser.add_argument(
        '-v', '--video',
        type=str,
        default='0',
        help="Path to video file. Leave empty to use your webcam."
    )
    args = parser.parse_args()

    # Determine if input is webcam index (0) or a file path
    source = int(args.video) if args.video.isdigit() else args.video

    detect_liveness(source)
