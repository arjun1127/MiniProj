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


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(254, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# see results using results.face_landmark.Landmark or and landmarks left hand or right hand
# results.landmarks for each face, pose, hands has its set of parameters x,y,z

# print(len(results.pose_landmarks.landmark)*4)
# print(len(results.face_landmarks.landmark)*3)
# print(len(results.left_hand_landmarks.landmark)*3)
# print(len(results.right_hand_landmarks.landmark)*3)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    # for slr concatenate all the keypoints
    return np.concatenate([pose, face, lh, rh])


# print(rh)
# print(extract_keypoints(results).shape) => (1662,)

# Path for exported data from the numpy array
DATA_PATH = os.path.join('MP_DATA')
# Actions that we try to detect
actions = np.array(['How are You', 'Im Fine', 'Hello', 'yes', 'ThankYou'])
# Thirty videos worth of data
no_sequences = 30
# videos are going to be 30 frames in length
sequence_length = 30

# MP_DATA/
# ├── hello/
# │   ├── seq_01/
# │   │   ├── 0.npy
# │   │   ├── 1.npy
# │   │   └── ...
# │   ├── seq_02/
# │   └── ...
# ├── thanks/
# │   ├── seq_01/
# │   └── ...
# └── iloveYou/
#     ├── seq_01/
#     └── ...

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# to capture the video by frames we have set up a loop

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Make detection
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                draw_landmarks(image, results)

                # Apply collection break for hand movement break
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} video number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Pause for 2 seconds
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} video number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # keypoints = extract_keypoints(results)
                # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # np.save(npy_path, keypoints)
                # Show the popup window and the img keeps on processing till while is true
                cv2.imshow('OpenCv Feed', image)

                # Quit the pop-up as you press 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            else:
                continue
            break
        else:
            continue
        break

result_test = extract_keypoints(results)
np.save('0', result_test)
np.load('0.npy')
