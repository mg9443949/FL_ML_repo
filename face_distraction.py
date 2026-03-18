import cv2
import mediapipe as mp
import numpy as np
import time
import math
#This code detects face distraction by monitoring if the face is in frame, if the eyes are closed, or if the head is turned away. It uses Mediapipe's Face Mesh to analyze facial landmarks and determine the user's focus level.
# -------------------- CONFIG --------------------
EAR_THRESHOLD = 0.20
HEAD_YAW_THRESHOLD = 25     # degrees
BUFFER_TIME = 5             # seconds
FPS = 30                    # approximate

# -------------------- MEDIAPIPE INIT --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- TIMERS --------------------
last_face_time = time.time()
eye_closed_time = 0
head_away_time = 0

# -------------------- EAR CALCULATION --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye, landmarks, w, h):
    p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    vertical1 = euclidean(p[1], p[5])
    vertical2 = euclidean(p[2], p[4])
    horizontal = euclidean(p[0], p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# -------------------- HEAD POSE (YAW) --------------------
def get_yaw_angle(landmarks, w, h):
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

    eye_mid = (left_eye + right_eye) / 2
    dx = nose[0] - eye_mid[0]
    yaw = (dx / w) * 100
    return yaw * 1.5

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    curr_time = time.time()
    delta_time = curr_time - prev_time
    prev_time = curr_time

    status = "Focused"

    # -------------------- FACE CHECK --------------------
    if not results.multi_face_landmarks:
        if curr_time - last_face_time > BUFFER_TIME:
            status = "Distraction: Face Not In Frame"
    else:
        last_face_time = curr_time
        landmarks = results.multi_face_landmarks[0].landmark

        # -------------------- DROWSINESS --------------------
        left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:
            eye_closed_time += delta_time
        else:
            eye_closed_time = 0

        if eye_closed_time > BUFFER_TIME:
            status = "Distraction: Drowsiness"

        # -------------------- HEAD TURN --------------------
        yaw = get_yaw_angle(landmarks, w, h)

        if abs(yaw) > HEAD_YAW_THRESHOLD:
            head_away_time += delta_time
        else:
            head_away_time = 0

        if head_away_time > BUFFER_TIME:
            status = "Distraction: Head Turned Away"

        # -------------------- DRAW FACE --------------------
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS
        )

    # -------------------- DISPLAY --------------------
    color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()