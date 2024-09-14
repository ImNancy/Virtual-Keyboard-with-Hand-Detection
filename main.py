import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)

# Keyboard settings
keys = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']]

# Variables
text = ""
keyboard_width = 1000
keyboard_height = 300
key_width = keyboard_width // 10
key_height = keyboard_height // 3


def draw_keyboard(img, alpha=0.3):
    overlay = img.copy()
    for i in range(3):
        for j, key in enumerate(keys[i]):
            x1, y1 = j * key_width, i * key_height
            x2, y2 = x1 + key_width, y1 + key_height
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.putText(overlay, key, (x1 + 20, y1 + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def get_clicked_key(x, y):
    row = y // key_height
    col = x // key_width
    if 0 <= row < 3 and 0 <= col < 10:
        return keys[row][col]
    return None


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (keyboard_width, keyboard_height))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    img = draw_keyboard(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * keyboard_width), int(index_finger_tip.y * keyboard_height)

            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

            clicked_key = get_clicked_key(x, y)
            if clicked_key:
                text += clicked_key
                print(f"Clicked: {clicked_key}")
                # Add a small delay to avoid multiple clicks
                cv2.waitKey(300)

    cv2.putText(img, text, (10, keyboard_height + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Virtual Keyboard", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()