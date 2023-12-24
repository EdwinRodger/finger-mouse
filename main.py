import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
index_y = 0
frame_height, frame_width = None, None
thumb_x, thumb_y = None, None

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if frame_height is None:
        frame_height, frame_width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            landmarks_np = np.array([[landmark.x, landmark.y] for landmark in landmarks])
            landmarks_np[:, 0] *= frame_width
            landmarks_np[:, 1] *= frame_height

            x = landmarks_np[:, 0].astype(int)
            y = landmarks_np[:, 1].astype(int)

            index_x = screen_width / frame_width * x[8]
            index_y = screen_height / frame_height * y[8]
            thumb_x = screen_width / frame_width * x[4]
            thumb_y = screen_height / frame_height * y[4]

            cv2.circle(img=frame, center=(x[8], y[8]), radius=10, color=(0, 255, 255))
            cv2.circle(img=frame, center=(x[4], y[4]), radius=10, color=(0, 255, 255))
            
            if abs(index_y - thumb_y) < 50:
                pyautogui.drag(x[4], y[4], button="left")

            pyautogui.moveTo(index_x, index_y)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()