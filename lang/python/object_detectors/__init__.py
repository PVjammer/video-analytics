import cv2
import logging
import numpy as np
import os
import requests
import sys
import time

from pydarknet import Detector, Image
# from vidstreamer import Streamer

DARKPY_PATH = os.path.join(sys.prefix, "darknet-files")
BLOCK_SIZE = 1024

darknet_config = {
    "yolov3": {
        "weights": os.path.join(DARKPY_PATH, "weights/yolov3.weights"),
        "config": os.path.join(DARKPY_PATH, "cfg/yolov3.cfg"),
        "data": os.path.join(DARKPY_PATH, "cfg/coco.data"),
        "weights_url": "https://pjreddie.com/media/files/yolov3.weights" 
    },
    "yolov3-tiny": {
        "weights": os.path.join(DARKPY_PATH, "weights/yolov3-tiny.weights"),
        "config": os.path.join(DARKPY_PATH, "cfg/yolov3-tiny.cfg"),
        "data": os.path.join(DARKPY_PATH, "cfg/coco.data"),
        "weights_url": "https://pjreddie.com/media/files/yolov3-tiny.weights"
    },
}

def download_model(model):
    r = requests.get(darknet_config[model]["weights_url"], stream=True)
    if not os.path.exists(os.path.join(DARKPY_PATH, "weights")):
        os.makedirs(os.path.join(DARKPY_PATH, "weights"))

    with open("darknet_config[model]['weights_url']", 'wb') as f:
        for data in r.iter_content(BLOCK_SIZE):
            f.write(data)
    logging.info("File {!s} successfully downloaded. Size: {!s}".format(darknet_config[model]["weights_url"], int(r.headers.get('content-length', 0))))

class Darknet:
    def __init__(self, model=None, config_path=None, weights_path=None, data_path=None):
        if not config_path and model:
            config_path = darknet_config[model]["config"]
        if not weights_path and model:
            weights_path = darknet_config[model]["weights"]
            if not os.path.exists(weights_path):
                download_model(model)
        if not data_path and model:
            data_path = darknet_config[model]["data"]

        
            
        
        self.net = Detector(bytes(config_path, encoding="utf-8"), bytes(weights_path, encoding="utf-8"), 0, 
                bytes(data_path, encoding="utf-8"))

    def detect(self, frame):
        dark_frame = Image(frame)
        results = self.net.detect(dark_frame)
        del dark_frame
        return results

