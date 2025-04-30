import cv2
#import sys
import numpy as np
source = cv2.VideoCapture(0)

cv2.namedWindow('camera preview' , cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe(r"models/deploy.prototxt",
                               r"models/res10_300x300_ssd_iter_140000.caffemodel")

in_width = 300
in_height = 300
mean= [104, 177, 123]

conf_threshold = 0.7

while cv2.waitKey(1) !=27:
    has_frame, frame = source.read()
    
    if not has_frame:
        break
    frame =cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence> conf_threshold:
            #x_top_left, y_top_left, x_bottom_right, y_bottom_right = int(detections[0,0,i,3:7] * [frame_width,frame_height,frame_width,frame_height])
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x_top_left, y_top_left, x_bottom_right, y_bottom_right) = box.astype("int")
            cv2.rectangle(frame, (x_top_left, y_top_left) , (x_bottom_right, y_bottom_right) , (0,255,0), 2)
            label = "Confidence %.3f"%confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(frame, (x_top_left, y_top_left - label_size[1]) , (x_top_left + label_size[0] , y_top_left + base_line) , (0,255,0) , cv2.FILLED)
            cv2.putText(frame, label, (x_top_left , y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    cv2.imshow("camera preview", frame)

source.release()
cv2.destroyAllWindows()        
            
            
            
            
            
