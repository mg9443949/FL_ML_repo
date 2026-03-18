import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from enum import Enum
# This code implements a hand gesture control system using Mediapipe's Hand Tracking. It detects specific hand gestures to control mouse movements, clicks, and scrolling on the computer. The gestures include moving the cursor, left-clicking, right-clicking, scrolling up, and scrolling down.
pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# smoothing
prev_x, prev_y = 0,0
smooth = 5

frame_margin = 100

click_delay = 0.4
last_click = 0


# -------------------- GESTURE ENUM --------------------

class Gesture(Enum):

    MOVE = 0
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    SCROLLUP = 3
    SCROLLDOWN = 4
    NONE = 5


# -------------------- FINGER DETECTION --------------------

def get_fingers_up(points):

    fingers = []

    # thumb
    fingers.append(points[4][0] > points[3][0])

    # index
    fingers.append(points[8][1] < points[6][1])

    # middle
    fingers.append(points[12][1] < points[10][1])

    # ring
    fingers.append(points[16][1] < points[14][1])

    # little
    fingers.append(points[20][1] < points[18][1])

    return fingers


# -------------------- GESTURE CLASSIFIER --------------------

def classify_gesture(fingers_up):

    # MOVE → all fingers up
    if fingers_up[1:5] == [ True, True, True, True]:
        return Gesture.MOVE

    # LEFT CLICK → thumb + index
    if fingers_up[1:5] == [True, False, False, False]:
        return Gesture.LEFT_CLICK

    # RIGHT CLICK → thumb + little
    if fingers_up[1:5] == [False, False, False, True]:
        return Gesture.RIGHT_CLICK


    # SCROLL UP → thumb only
    if fingers_up == [True, False, False, False, False]:
        return Gesture.SCROLLUP

    # SCROLL DOWN → fist
    if fingers_up == [False, False, False, False, False]:
        return Gesture.SCROLLDOWN

    return Gesture.NONE


# -------------------- MAIN LOOP --------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    cv2.rectangle(frame,
                  (frame_margin,frame_margin),
                  (w-frame_margin,h-frame_margin),
                  (0,255,0),2)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(frame,
                               hand,
                               mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark

        points = []

        for i in range(21):

            x = int(lm[i].x*w)
            y = int(lm[i].y*h)

            points.append((x,y))

        index = points[8]

        # detect fingers
        fingers = get_fingers_up(points)

        # classify gesture
        gesture = classify_gesture(fingers)


        # ---------------- MOVE ----------------

        if gesture == Gesture.MOVE:

            screen_x = np.interp(index[0],
                                 (frame_margin,w-frame_margin),
                                 (0,screen_w))

            screen_y = np.interp(index[1],
                                 (frame_margin,h-frame_margin),
                                 (0,screen_h))

            curr_x = prev_x + (screen_x-prev_x)/smooth
            curr_y = prev_y + (screen_y-prev_y)/smooth

            pyautogui.moveTo(curr_x,curr_y)

            prev_x,prev_y = curr_x,curr_y


        # ---------------- LEFT CLICK ----------------

        elif gesture == Gesture.LEFT_CLICK:

            if time.time()-last_click > click_delay:

                pyautogui.click()

                last_click = time.time()


        # ---------------- RIGHT CLICK ----------------

        elif gesture == Gesture.RIGHT_CLICK:

            pyautogui.rightClick()


        # ---------------- SCROLL UP ----------------

        elif gesture == Gesture.SCROLLUP:

            pyautogui.scroll(50)


        # ---------------- SCROLL DOWN ----------------

        elif gesture == Gesture.SCROLLDOWN:

            pyautogui.scroll(-50)


        cv2.putText(frame,
                    str(gesture.name),
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)


    cv2.imshow("AI Gesture Mouse",frame)

    if cv2.waitKey(1)==27:
        break


cap.release()
cv2.destroyAllWindows()