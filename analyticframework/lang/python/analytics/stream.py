import cv2 as cv
import numpy as np
import sys
from object_detector.objectdetection import ObjectDetector

cap  = cv.VideoCapture(0)
# print(cap.isOpened())
if not cap.isOpened():
    print("Cannot capture video")
    sys.exit(1)

face_cascade = cv.CascadeClassifier('/usr/local/Cellar/opencv/4.0.1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/usr/local/Cellar/opencv/4.0.1/share/opencv4/haarcascades/haarcascade_eye.xml')
detector = ObjectDetector(model_path="object_detector/models/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb")

class PhoneFinder:

    def __init__():
        self.is_phone = False

    def check_for_phone(classes):
        if self.is_phone and "cellphone" not in classes.lower():
            self.is_phone = False
        elif not self.isphone and "cellphone" in classes.lower():
            self.is_phone = True
            print("FOUND CELLPHONE")


while True:
    ret, frame = cap.read()

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame, detection_dict = detector.detect(frame)
    print(detection_dict["detection_classes"])
    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

    except Exception as e:
        print(e)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
