import cv2
from collections import Counter
from modules import find_position, find_landmark_name, speak
import math

capture = cv2.VideoCapture(0)
tip = [4, 8, 12, 16, 20]
prev_interpreted_word = 0
finger_count = []
fingers = []

while True:
    ret, frame = capture.read()

    # Resizing the frame is necessary to reduce the processing load for the model
    RESIZED_WIDTH = 640
    RESIZED_HEIGHT = 360
    resized_frame = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))

    landmark_positions = find_position(resized_frame)
    landmark_names = find_landmark_name(resized_frame)

    if len(landmark_positions) != 0:
        finger_count = []
        if landmark_positions[0][1] < landmark_positions[4][1]:
            finger_count.append(1)

        else:
            finger_count.append(0)

        finfinger_countger = []
        for id in range(0, 4):
            if landmark_positions[tip[id]][1:] < landmark_positions[tip[id] - 2][1:]:
                fingers.append(1)

    x = fingers + finger_count
    c = Counter(x)
    interpreted_word = c[1]

    cv2.imshow("Frame", resized_frame)
    key = cv2.waitKey(1) & 0xFF

    if interpreted_word != prev_interpreted_word:
        speak(str(interpreted_word))
        prev_interpreted_word = interpreted_word

    if key == ord("q"):
        break

    print("Position", landmark_positions)

capture.release()
cv2.destroyAllWindows()
