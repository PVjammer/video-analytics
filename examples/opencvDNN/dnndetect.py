import cv2
import numpy as np
import json
import os

from vidstreamer import Streamer, analytic_pb2, StreamerParam
from object_detectors import DNNDetector

# backends = (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv2.dnn.DNN_BACKEND_OPENCV)
# targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)

MODEL_PATH = "models"
confThreshold = 0.5
model = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
config = os.path.join(MODEL_PATH, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")


with open("classes.json", 'r') as f:
    classes = json.load(f)

def process(img, req, resp):
    resp = detector.detect(img, req, resp)

def load_model(streamer):
    global detector
    print(streamer.params)
    # net = cv2.dnn.readNetFromTensorflow(streamer.params["model_path"], streamer.params["config_path"])
    detector = DNNDetector(model_path=streamer.params["model_path"], config_path=streamer.params["config_path"], classes=classes, threshold=streamer.params["threshold"])

if __name__ == "__main__":
    detector = None
    streamer = Streamer(func=process, output_func="render")

    # Analytic specific parameters
    params = []
    params.append(StreamerParam(name="--model_path", default="models/frozen_inference_graph.pb", type=str, helptext="Path of the model to load"))
    params.append(StreamerParam(name="--config_path", default="models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", type=str, helptext="Path of the config file to load"))
    params.append(StreamerParam(name="--threshold", default=0.5, helptext="Detection Threshold"))

    try:
        streamer.run(params, init_func=load_model)
    finally:
        cv2.destroyAllWindows()
