# Real-Time Face Detector
This project performs real-time face detection using OpenCV's Deep Neural Network (DNN) module and a pre-trained SSD model based on the Caffe framework.

## Features
* Real-time face detection from webcam

* Uses OpenCV DNN with res10_300x300_ssd_iter_140000.caffemodel

* Displays bounding boxes with confidence scores

## Requirements
* Python 3.x

* OpenCV (4.x recommended)

* NumPy

## How to Run
1. Clone this repository or download the script.

2. Download the model files:

* deploy.prototxt
(https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)

* res10_300x300_ssd_iter_140000.caffemodel
(https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

3. Run the script using Python (e.g., python face_detection.py).

4. Press Esc to close the window.