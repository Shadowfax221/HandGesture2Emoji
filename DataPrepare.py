import csv
import json
import os
import random

import mediapipe as mp
import numpy as np

LANDMARKER_MODEL_PATH = 'models/hand_landmarker.task'
CSV_DATASET_PATH = 'datasets/HandLandmarks.csv'
IMAGE_DATASET_DIR = "E:/MyDatasets/hagrid_dataset_512"
ANNOTATIONS_DIR = "E:/MyDatasets/hagrid_dataset_annotations/train"
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandDataPrepare():
    def __init__(self):
        self.NUM_SAMPLES = 125

    # Pop out a random key
    def pop_random_key(self, keys_list):
        if not keys_list:
            print("No more keys to select.")
            return None
        random_idx = random.randint(0, len(keys_list) - 1)
        return keys_list.pop(random_idx)

    
    # Check if the point (x, y) is within the bounding box.
    def is_point_in_bbox(self, x, y, bbox, margin=0.01):
        if bbox == None:
            return True
        tl_x, tl_y, width, height = bbox
        ext_tl_x = tl_x - margin
        ext_tl_y = tl_y - margin
        ext_br_x = tl_x + width + margin
        ext_br_y = tl_y + height + margin
        return ext_tl_x <= x <= ext_br_x and ext_tl_y <= y <= ext_br_y
    

    # Preprocess the hand landmark
    def pre_process_landmark(self, hand_landmarks, handedness, gesture_bboxes=None):
        landmark_list = []
        # Convert to relative coordinates
        for idx, landmark in enumerate(hand_landmarks):
            if self.is_point_in_bbox(landmark.x, landmark.y, gesture_bboxes):
                if idx == 0:
                    base_x, base_y = landmark.x - 0.5, landmark.y - 0.5
                    landmark_list.extend([0.5, 0.5, 0])
                else:
                    landmark_list.extend([landmark.x - base_x, landmark.y - base_y, landmark.z])
            else:
                break
        # Convert to numpy array and add handedness
        landmark_array = np.array([handedness[0].index] + landmark_list).astype(np.float32)         # Right is 0, Left is 1
        return landmark_array


    def main(self):
        # STEP 1: Create an HandLandmarker object.
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE)
        
        with HandLandmarker.create_from_options(options) as landmarker:
            with open(CSV_DATASET_PATH, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)

                # STEP 2: Load the input image.
                for label in LABELS:
                    with open(os.path.join(ANNOTATIONS_DIR, f"{label}.json"), 'r') as file:
                        annotations = json.load(file)
                    annotations_keys = list(annotations.keys())
                    
                    samples_cnt = 0
                    while samples_cnt < self.NUM_SAMPLES:
                        image_name = self.pop_random_key(annotations_keys)
                        image_path = os.path.join(IMAGE_DATASET_DIR, label, f'{image_name}.jpg')
                        mp_image = mp.Image.create_from_file(image_path)

                        # STEP 3: Detect hand landmarks from the input image.
                        hand_landmarker_result = landmarker.detect(mp_image)

                        if hand_landmarker_result is not None:
                            # STEP 4: Write hand landmark into csv
                            annotations_labels = annotations[image_name]['labels']
                            annotations_bboxes = annotations[image_name]['bboxes']
                            gesture_bboxes = annotations_bboxes[annotations_labels.index(label)]

                            for hand_landmarks, handedness in zip(hand_landmarker_result.hand_landmarks,
                                                                  hand_landmarker_result.handedness):
                                landmark_array = self.pre_process_landmark(hand_landmarks, handedness, gesture_bboxes)
                                row = [LABELS.index(label)] + landmark_array.tolist()
                                
                                if len(row) == 65:
                                    csvwriter.writerow(row)
                                    samples_cnt += 1
                    print(f"Writen {samples_cnt} samples of {label}")



if __name__ == "__main__":
    body_module = HandDataPrepare()
    body_module.main()