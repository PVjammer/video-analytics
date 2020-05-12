import argparse
import cv2
import numpy as np


def load_model(prototext, model):
    return cv2.dnn.readNetFromCaffe(prototext, model)

def get_blob(img):
    return cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

def get_boxes(detections, threshold, w, h):
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence < threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        boxes.append(box.astype("int"))
    return boxes

def draw_boxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
    return img
    


def main(args):
    model = load_model(args.proto, args.model)
    img = cv2.imread(args.img)
    blob = get_blob(img)
    model.setInput(blob)
    detections = model.forward()
    (h, w) = img.shape[:2]
    boxes = get_boxes(detections, args.threshold, w, h)
    img = draw_boxes(img, boxes)

    cv2.imshow("Face image", img)
    cv2.waitKey(0)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Image file to process" )
    parser.add_argument("--proto", "-p", default="models/deploy.prototxt", help="Path to caffe prototxt file")
    parser.add_argument("--model", "-m", default="models/res10_300x300_ssd_iter_140000_fp16.caffemodel", help="Model path")
    parser.add_argument("--threshold", "-t", help="Threshold at which to say a face was detected", type=float, default=0.6)

    args = parser.parse_args()
    main(args)

