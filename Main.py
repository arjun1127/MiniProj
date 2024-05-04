# this is the main file where we'll be importing the modules

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

# to capture the video by frames we have set up a loop

cap = cv2.VideoCapture(0)
while cap.isOpened():
    # to read the frame
    ret, frame = cap.read()
    # to show the popup window and the img keeps on processing till while is true
    cv2.imshow('OpenCv Feed', frame)
    # quit the pop up as u press 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
