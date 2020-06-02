import numpy as np
import os
import time
import sys
from ace import analyticservice, analytic_pb2

DARKPY_PATH = "."
from pydarknet import Detector, Image
import cv2 as cv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFG = os.path.join(DARKPY_PATH, "cfg", "yolov3-tiny.cfg")
WEIGHTS = os.path.join(DARKPY_PATH, "weights", "yolov3-tiny.weights")
DATA = os.path.join(DARKPY_PATH, "cfg", "coco.data")

def process_frame(req, resp):
    """ Process individual video frames"""

    img_bytes = np.fromstring(req.frame.img, dtype=np.uint8)
    frame = cv.imdecode(img_bytes,1)
    dark_frame = Image(frame)
    results = net.detect(dark_frame)
    del dark_frame

    for cat, score, bounds in results:
        x, y, w, h = bounds
    

        rect = analytic_pb2.RegionOfInterest()
        rect.confidence = score
        print("{!s} - {!s}".format(cat, score))
        box = analytic_pb2.BoundingBox(corner1=analytic_pb2.Point(x=int(x-w/2), y=int(y-h/2)),
                                       corner2 = analytic_pb2.Point(x=int(x+w/2), y=int(y+h/2)))
        rect.box.MergeFrom(box)
        rect.classification = str(cat.decode("utf-8"))
        resp.data.roi.extend([rect])


if __name__ == "__main__":
    # Optional statement to configure preferred GPU. Available only in GPU version.
    import argparse
    import pydarknet
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="Host of the proxy")
    parser.add_argument("--port", default=50051, help="Port the proxy will run on.")

    args = parser.parse_args()

    # pydar knet.set_cuda_device(0)
    dknet_config ={
        "cfg_path" : CFG,
        "weights_path" : WEIGHTS,
        "data_path" : DATA
    }
    logger.info(dknet_config)
    net = Detector(bytes(dknet_config["cfg_path"], encoding="utf-8"), bytes(dknet_config["weights_path"], encoding="utf-8"), 0, bytes(dknet_config["data_path"], encoding="utf-8"))

    svc = analyticservice.AnalyticService()
    svc.register_name("Yolo v3")
    svc.RegisterProcessVideoFrame(process_frame)
    sys.exit(svc.Run(analytic_port=int(args.port)))


