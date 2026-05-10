import cv2
import mediapipe as mp
import math
import time
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

def detect_liveness():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    blink_counter = 0
    start_time = time.time()

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

                    if left_ear < 0.21:
                        blink_counter += 1

                elapsed = time.time() - start_time
                if elapsed > 3.0:  # Check blinks over a period
                    if blink_counter > 5:
                        liveness_status = "Status: Live (Blinks Detected)"
                        color = (0, 255, 0)
                    else:
                        liveness_status = "Status: Fake/Spoof Warning!"
                        color = (0, 0, 255)

            cv2.putText(frame, liveness_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "Press 'q' to quit", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Liveness Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_liveness()
