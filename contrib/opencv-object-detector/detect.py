import cv2
import numpy as np
import json
import os

from vidstreamer import Streamer, analytic_pb2

# backends = (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv2.dnn.DNN_BACKEND_OPENCV)
# targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)

MODEL_PATH = "models"
confThreshold = 0.5
model = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
config = os.path.join(MODEL_PATH, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")


with open("classes.json", 'r') as f:
    classes = json.load(f)

def process(img, req, resp):
    img_ht, img_width, _ = img.shape
    net.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True))
    output = net.forward()
    
    for detection in output[0, 0, :, :]:
        roi = analytic_pb2.RegionOfInterest()
        roi.confidence = detection[2]
        if roi.confidence > confThreshold:
            roi.classification = classes[str(int(detection[1]))]
            bounding_box = analytic_pb2.BoundingBox(
                corner1=analytic_pb2.Point(x=int(detection[3] * img_width), y=int(detection[4] * img_ht)),
                  corner2=analytic_pb2.Point(x=int(detection[5] * img_width), y=int(detection[6] * img_ht)))  
            roi.box.MergeFrom(bounding_box)
            resp.roi.extend([roi])


if __name__ == "__main__":
    net = cv2.dnn.readNetFromTensorflow(model, config)
    streamer = Streamer(func=process, output_func="render")
    # streamer.register_output_func(render)
    try:
        streamer.run()
    finally:
        cv2.destroyAllWindows()
