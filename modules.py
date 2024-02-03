import cv2
import mediapipe as piper
import matplotlib.pyplot as pyplot
# import OS
import pyttsx3

drawingModule = piper.solutions.drawing_utils
handsModule = piper.solutions.hands
mod = handsModule.Hands()

RESIZED_WIDTH = 640
RESIZED_HEIGHT = 360


def speak(text):
    engine = pyttsx3.init()
    print(type(text))
    engine.say(text)
    engine.runAndWait()


def find_position(frame):
    node_list = []
    results = mod.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(
                frame, handLandmarks, handsModule.HAND_CONNECTIONS
            )
            node_list = []
            for id, node in enumerate(handLandmarks.landmark):
                x = int(RESIZED_WIDTH - (node.x * RESIZED_WIDTH))
                y = int(RESIZED_HEIGHT - (node.y * RESIZED_HEIGHT))
                node_list.append([id, x, y])

    return node_list


def find_landmark_name(frame):
    list = []
    results = mod.process(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            for point in handsModule.HandLandmark:
                list.append(str(point).replace("< ", "").replace(
                    "HandLandmark.", "").replace("_", " ").replace("[]", "")
                )
    return list
