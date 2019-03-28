import os
import cv2 as cv
import argparse
import logging
import numpy as np
import tensorflow as tf
import sys

sys.path.append(os.path.join(os.environ['TF_MODEL_PATH'], "research"))
import object_detector.obj_util as obj_util

sys.path.append(os.environ['TF_MODEL_PATH'])
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ObjectDetector:

    def __init__(self, model_path='models/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb',
                    label_path=os.path.join(obj_util.DEFAUALT_LABEL_PATH, 'mscoco_label_map.pbtxt')):
        log.info("Instantiating object detector")
        self.graph = obj_util.load_model(model_path)
        self.category_index = obj_util.load_label_map(label_path)
        self.sess = tf.Session(graph=self.graph)
        self.model = model_path
        # print(self.category_index)
        log.info("Model instantiated")

    def detect(self, img):

        img_exp = np.expand_dims(img, axis=0)
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: img_exp})

        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return img, {"bounding_boxes":boxes, "detection_scores": scores,
                    "detection_classes": classes, "num_detections": num_detections}

if __name__ == "__main__":
    obj=ObjectDetector()
    # cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
    # detector = ObjectDetector()
    # print("Model loaded")
    # while True:
    #
    #     ret, frame = cap.read()
    #     img, detection_dict = detector.detect(frame)
    #     cv.imshow('frame', img)
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv.destroyAllWindows()
