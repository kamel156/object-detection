import time

import cv2
import numpy as np
import timer

# import video
cap = cv2.VideoCapture('traffic.mp4')
counter = 0
#Load Yolo net
net = cv2.dnn.readNet('yolov3tiny.cfg', 'yolov3-tiny.weights')
classes = []
with open('cocco.names', 'r') as f:
    classes =[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))

while True:
    success, frame = cap.read()
    height, width, channels = frame.shape

    #Detecting Objects

    blob = cv2.dnn.blobFromImage(frame, 0.00392,(320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    #Print info on screen

    class_ids  =[]
    confidences = []
    boxes =[]
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > .5:
                center_x =int(detection[0]* width)
                center_y = int(detection[1] * height)
                w = int(detection[2]* width)
                h = int(detection[2] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'bus':
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0),2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0),3)

                counter+=1
    print(counter)
    cv2.imshow('image' , frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()