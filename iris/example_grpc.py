import cv2 as cv
import logging
import numpy as np
import sys
import os
import argparse

#Todo: delete after testing
import random

from ace import analyticservice, analytic_pb2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

har_path = "./haarfiles"
face_cascade = cv.CascadeClassifier(os.path.join(har_path,'haarcascade_frontalface_default.xml'))
eye_cascade = cv.CascadeClassifier(os.path.join(har_path,'haarcascade_eye.xml'))


def detect(req, resp):

    img_bytes = np.fromstring(req.frame.img, dtype=np.uint8)
    frame = cv.imdecode(img_bytes,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        rect = analytic_pb2.RegionOfInterest()
        box = analytic_pb2.BoundingBox(corner1=analytic_pb2.Point(x=x, y=y), 
                                       corner2 = analytic_pb2.Point(x=x+w, y=y+h))
        rect.box.MergeFrom(box)
        rect.classification = "Face"
        #Todo: delete after testing
        rect.confidence = random.random()
        resp.data.roi.extend([rect])

    for (x,y,w,h) in eyes:
        rect = analytic_pb2.RegionOfInterest()
        box = analytic_pb2.BoundingBox(corner1=analytic_pb2.Point(x=x, y=y), 
                                       corner2 = analytic_pb2.Point(x=x+w, y=y+h))
        rect.box.MergeFrom(box)
        rect.classification = "Eye"
        #Todo: delete after testing
        rect.confidence = random.random()
        resp.data.roi.extend([rect])
    
    print("Found {!s} faces".format(len(faces)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="Host of the proxy")
    parser.add_argument("--port", default=50051, help="Port the proxy will run on.")

    args = parser.parse_args()
    svc = analyticservice.AnalyticService()
    svc.register_name("Haar Detector")
    svc.RegisterProcessVideoFrame(detect)
    sys.exit(svc.Run(analytic_port=int(args.port)))
