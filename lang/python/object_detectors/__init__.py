import cv2
import logging
import numpy as np
import os
import sys
import time

from pydarknet import Detector, Image
# from vidstreamer import Streamer

DARKPY_PATH = "./darknet"

CFG = os.path.join(DARKPY_PATH, "cfg", "yolov3-tiny.cfg")
WEIGHTS = os.path.join(DARKPY_PATH, "weights", "yolov3-tiny.weights")
DATA = os.path.join(DARKPY_PATH, "cfg", "coco.data")

class Darknet:
    def __init__(self, config_path=CFG, weights_path=WEIGHTS, data_path=DATA):
        self.net = Detector(bytes(config_path, encoding="utf-8"), bytes(weights_path, encoding="utf-8"), 0, 
                bytes(data_path, encoding="utf-8"))

    def detect(self, frame):
        dark_frame = Image(frame)
        results = self.net.detect(dark_frame)
        del dark_frame
        return results

