import cv2
import numpy as np



from vidstreamer import Streamer, analytic_pb2

class DNNDetector:
    def __init__(self, model_path=None, config_path=None, classes=None, threshold=0.5, cnn_size=(300, 300)):
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.cnn_size = cnn_size          
        self.threshold = threshold
        self.classes = classes

    def detect(self, img, req=None, resp=None):
        img_ht, img_width, _ = img.shape
        self.net.setInput(cv2.dnn.blobFromImage(img, size=self.cnn_size, swapRB=True))
        output = self.net.forward()

        for detection in output[0, 0, :, :]:
            roi = analytic_pb2.RegionOfInterest()
            roi.confidence = detection[2]
            if roi.confidence > self.threshold:
                roi.classification = self.classes[str(int(detection[1]))]
                bounding_box = analytic_pb2.BoundingBox(
                    corner1=analytic_pb2.Point(x=int(detection[3] * img_width), y=int(detection[4] * img_ht)),
                    corner2=analytic_pb2.Point(x=int(detection[5] * img_width), y=int(detection[6] * img_ht)))  
                roi.box.MergeFrom(bounding_box)
                resp.roi.extend([roi])
