# this is the main file where we'll be importing the modules

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting image from BGR TO RGB (for Ml and mediaPie detection)
    image.flags.writeable = False  # this ensures that data image arr is not modified before processing
    results = model.process(image)  # method provided by the framework to
    # apply a trained model to input data and generate predictions or outputs.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # for o/p image to be in correct format
    return image, results


# to capture the video by frames we have set up a loop

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # to read the frame
        ret, frame = cap.read()
        # make detection
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        # to show the popup window and the img keeps on processing till while is true
        cv2.imshow('OpenCv Feed', frame)
        # quit the pop-up as u press 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# see results using results.face_landmark.Landmark or and landmarks left hand or right hand

