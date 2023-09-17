import cv2
import torch
import numpy as np
from imutils.video import VideoStream
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture('in.avi')  # Change this to your video file path or 0 for default webcam
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Natural Disaster Compilation.mp4')
# cap = VideoStream(src=0).start()
# Camera intrinsic parameters (you need to calibrate your camera and get these values)
focal_length = 800  # Focal length in pixels
object_width = 20  # cm

def calculate_distance(bbox_width):
    distance = (focal_length * object_width) / bbox_width
    return distance

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = model(frame)
    pred = results.pred[0]
    # print(pred)
    for det in pred:
        label, conf, bbox = det[5], det[4], det[:4]
        if label == 0 and conf>=0.60:  # Assuming class index 0 corresponds to humans
            x1, y1, x2, y2 = map(int, bbox)
            obj_width = x2-x1
            bbox_height = y2 - y1
            distance = calculate_distance(obj_width)/100
            print("Distance: ",round(distance)," m.")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Human: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f'Distance: {distance:.2f} m.', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)                                                                                                     

    # Display the frame
    cv2.imwrite("detection.mp4",img)
    cv2.imshow('Human Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
