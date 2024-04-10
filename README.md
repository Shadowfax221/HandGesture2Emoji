# HandGesture2Emoji

This program provides demo of live stream hand gesture classification. I contains 3 parts: 1) dataset selection, 2) model training, 3) live recogniation demo. 

The dataset is selected from [HaGRID - HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid), which contains the hand gestures below:
![gestures](https://github.com/hukenovs/hagrid/raw/master/images/gestures.jpg)

In dataset selection, we select part of hand gestures, and match each to an emoji.
| idx | dataset     | emoji         | shortcode                 | Unicode     | comment |
|-----|-------------|---------------|---------------------------|-------------|---------|
|0| call        | 🤙            | `:call_me_hand:`          | U+1F919     |         |
|1| dislike     | 👎            | `:thumbs_down:`           | U+1F44E     |         |
|2| fist        | ✊            | `:raised_fist:`           | U+270A      |         |
|3| like        | 👍            | `:thumbs_up:`             | U+1F44D     |         |
|4| mute        | 🤐            | `:zipper_mouth_face:`     | U+1F910     | uncommon |
|5| ok          | 👌            | `:ok_hand:`               | U+1F44C     |         |
|6| one         | ☝             | `:index_pointing_up:`     | U+261D      |         |
|7| palm        | 🖐            | `:raised_hand_with_fingers_splayed:` | U+1F590 |         |
|8| peace       | ✌             | `:victory_hand:`          | U+270C      |         |
|9| rock        | 🤘            | `:sign_of_the_horns:`     | U+1F918     |         |
|10| stop        | ✋            | `:raised_hand:`           | U+270B      | similar to stop inv        |
|11| stop inv.   | 🤚            | `:raised_back_of_hand:`   | U+1F91A     | similar to stop        |


We prepare the training dataset into `HandLandmarks.csv`. The process is done by `DataPrepare.py`. We first detection the hand landmarks of images in HaGRID dataset by [mediapipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) landmarker model. Each hand landmark will be preprocessed and converted into one row (65 columns) in csv file. The structure of one row is: 1) index 0 is label, same as the table above; 2) index 1 is handerness, right is 0 and left is 1; 3) index 2 to index 64 are positions (x,y,z) of 21 landmarks of a hand, shown in image below. The hand landmarks will be preprocessed, steps are: 1) shift all landmark points so point0 is at (0,0,0) 2) rotate all (x,y) to set the line between point0 and point5 vertical. 
![](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)


The model is trained by `gesture_classification.ipynb`, which output a model `gesture_classifier.tflite`. The demo `gesture_classifier.py` use the model to preform live stream hand gesture classification. 
