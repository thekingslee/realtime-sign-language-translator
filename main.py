#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
from collections import Counter
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

from modules import get_args, calc_bounding_rect, select_mode, calc_landmark_list, pre_process_landmark, draw_landmarks, logging_csv, pre_process_point_history, draw_info, draw_point_history, draw_info_text, draw_bounding_rect, speak


def main():
    # Argument parsing #################################################################
    args = get_args()

    capture_device = args.device
    capture_width = args.width
    capture_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    inferred_meaning = ""
    delayed_counter = 0

    # Camera preparation ###############################################################
    capture = cv2.VideoCapture(capture_device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = capture.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    ""
                )
                if (inferred_meaning != keypoint_classifier_labels[hand_sign_id]):
                    if (delayed_counter == 15):
                        speak(str(keypoint_classifier_labels[hand_sign_id]))
                        delayed_counter = 0
                        inferred_meaning = keypoint_classifier_labels[hand_sign_id]
                    else:
                        delayed_counter += 1

        else:
            point_history.append([0, 0])

        debug_image = draw_info(debug_image, delayed_counter, mode, number)

        # Screen reflection #############################################################
        cv2.imshow('Hand Gesture Recognition', debug_image)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
