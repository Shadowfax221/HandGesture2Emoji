import csv
import json
import os
import random

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_SAMPLES = 100
DATASET_DIR = "E:/MyDatasets/hagrid_dataset_512"
ANNOTATIONS_DIR = "C:/Users/Ian/git/553.806_Capstone_HandGesture/annotations/test"
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']     # 12 gestures: 🤙, 👎, ✊, 👍, 🤐, 👌, ☝, 🖐, ✌, 🤘, ✋, 🤚


# Check if the point (x, y) is within the bounding box.
def is_point_in_bbox(x, y, bbox, margin=0.01):
    tl_x, tl_y, width, height = bbox
    ext_tl_x = tl_x - margin
    ext_tl_y = tl_y - margin
    ext_br_x = tl_x + width + margin
    ext_br_y = tl_y + height + margin
    return ext_tl_x <= x <= ext_br_x and ext_tl_y <= y <= ext_br_y



# STEP 1: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

csv_filename = 'keypoint.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # STEP 2: Load the input image.
    for label in LABELS:
        with open(os.path.join(ANNOTATIONS_DIR, f"{label}.json"), 'r') as file:
            annotations = json.load(file)
        
        samples_cnt = 0
        while samples_cnt < NUM_SAMPLES:
            image_name = random.choice(list(annotations.keys()))
            image_path = os.path.join(DATASET_DIR, label, f'{image_name}.jpg')
            image = mp.Image.create_from_file(image_path)

            # STEP 3: Detect hand landmarks from the input image.
            detection_result = detector.detect(image)

            # STEP 4: Write hand landmark into csv
            annotations_labels = annotations[image_name]['labels']
            annotations_bboxes = annotations[image_name]['bboxes']
            gesture_bboxes = annotations_bboxes[annotations_labels.index(label)]

            label_dict = {label: i for i, label in enumerate(LABELS)}
            for hand_landmarks in detection_result.hand_landmarks:
                row = [label_dict[label]] 
                for landmark in hand_landmarks:
                    if is_point_in_bbox(landmark.x, landmark.y, gesture_bboxes):
                        row.extend([landmark.x, landmark.y, landmark.z])        # may add landmark.z
                
                if len(row) == 64:
                    csvwriter.writerow(row)
                    samples_cnt += 1
        print(f"Writen {samples_cnt} samples of {label}")

print("CSV file has been created:", csv_filename)
                    