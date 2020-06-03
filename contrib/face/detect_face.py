import cv2
import numpy as np

from vidstreamer import Streamer, analytic_pb2

THRESHOLD = 0.6

def load_model(prototext, model):
    return cv2.dnn.readNet(prototext, model)

def get_blob(img):
    return cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

def get_boxes(detections, threshold, w, h):
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence < threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        box = box.astype("int")
        bounding_box = analytic_pb2.BoundingBox(corner1=analytic_pb2.Point(x=box[0], y=box[1]),
                  corner2=analytic_pb2.Point(x=box[2], y=box[3]))

        roi = analytic_pb2.RegionOfInterest()
        roi.confidence = confidence
        roi.classification = "Face"
        roi.box.MergeFrom(bounding_box)
        boxes.append(roi)
    return boxes

def process(frame, req, resp):
    blob = get_blob(frame)
    model.setInput(blob)
    detections = model.forward()
    (h, w) = frame.shape[:2]
    boxes = get_boxes(detections, THRESHOLD, w, h)
    resp.roi.extend(boxes)
    


if __name__ == "__main__":
    model = load_model("models/deploy_lowres.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    streamer = Streamer(func=process, output_func="render")
    # streamer.register_output_func(render)
    try:
        streamer.run()
    finally:
        cv2.destroyAllWindows()