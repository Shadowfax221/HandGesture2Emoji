import copy
import itertools

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

LANDMARKER_MODEL_PATH = 'models/hand_landmarker.task'
CLASSIFIER_MODEL_PATH = 'models/gesture_classifier_1000original.tflite'
LABELS = ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'stop_inverted']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class HandLiveRecognition():
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None


    def pre_process_landmark_original(self, hand_landmarks, handedness):
        landmark_list = []
        for _, landmark in enumerate(hand_landmarks):
            landmark_list.extend([landmark.x, landmark.y, landmark.z])
        # Convert to numpy array and add handedness
        landmark_array = np.array([handedness[0].index] + landmark_list).astype(np.float32)         # Right is 0, Left is 1
        return landmark_array
    

    def load_tflite_model(self, tflite_model_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        return interpreter

    def tflite_predict(self, model, input_data):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], np.array([input_data]))
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        return output_data

    def integrated_prediction(self, model, input_data):
        pred = self.tflite_predict(model, input_data)
        scores = np.max(pred, axis=1)
        label = np.argmax(pred, axis=1)
        return scores, label


    def draw_landmarks_on_image(self, annotated_image, hand_landmarks):
        self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(255, 0, 255), thickness=4, circle_radius=2),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(20, 180, 90), thickness=2, circle_radius=2)
        )
        return annotated_image
    

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def draw_bounding_rect(self, annotated_image, brect):
        cv2.rectangle(annotated_image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 1)
        return annotated_image
    

    def draw_info_text(self, image, brect, handedness, score, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)
        handedness_text = handedness[0].display_name
        info_text = f"{handedness_text}:{score:.2f}"
        if hand_sign_text != "":
            info_text = hand_sign_text + ':' + info_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return image



    # Create a gesture landmarker instance with the live stream mode:
    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def main(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.print_result)
        
        capture = cv2.VideoCapture(0)

        timestamp = 0
        with HandLandmarker.create_from_options(options) as landmarker:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    print("Ignoring empty frame")
                    break
                frame = cv2.flip(frame, 1)
                annotated_image = copy.deepcopy(frame)
                
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, timestamp)

                if self.results is not None:
                    for hand_landmarks, handedness in zip(self.results.hand_landmarks,
                                                          self.results.handedness):
                        # hand_landmarks set up ########################################################
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                        ])

                        # Hand Classification ##########################################################
                        pre_processed_landmarks = self.pre_process_landmark_original(hand_landmarks, handedness)
                        model = self.load_tflite_model(CLASSIFIER_MODEL_PATH)
                        score, predictions = self.integrated_prediction(model, pre_processed_landmarks)
                        
                        # drawing part #################################################################
                        annotated_image = self.draw_landmarks_on_image(annotated_image, hand_landmarks_proto)
                        brect = self.calc_bounding_rect(annotated_image, hand_landmarks_proto)
                        annotated_image = self.draw_bounding_rect(annotated_image, brect)
                        annotated_image = self.draw_info_text(
                            annotated_image,
                            brect,
                            handedness,
                            score[0],
                            LABELS[predictions[0]],
                        )
                        # print(self.results.gestures)
                        
                    cv2.imshow('Show', annotated_image)
                else:
                    cv2.imshow('Show', frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")
                    break
                        
            capture.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    body_module = HandLiveRecognition()
    body_module.main()