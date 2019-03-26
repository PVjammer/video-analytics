import numpy as np
import os
import six.moves.urllib as urllib
import shutil
import sys
import tarfile
import tensorflow as tf

sys.path.append("/Users/nicholasburnett/Workspace/ace/video-analytics/external-packages/tensorflow/models/research/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from io import StringIO

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEFAUALT_LABEL_PATH = '/Users/nicholasburnett/Workspace/ace/video-analytics/external-packages/tensorflow/models/research/object_detection/data/'

def load_label_map(label_path=os.path.join(DEFAUALT_LABEL_PATH, 'mscoco_label_map.pbtxt')):
    return label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

def download_model(model_file='ssd_inception_v2_coco_11_06_2017.tar.gz', dst="models"):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + model_file, model_file)
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            if not os.path.exists(dst):
                os.makedirs(dst)
            tar_file.extract(file, dst)
    os.remove(model_file)

def load_model(graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path,'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def wrap_viz(image, boxes, classes, scores, category_index, use_normalized_coordinates=True, line_thickness=8):

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        line_thickness=line_thickness)
    return image

if __name__ == "__main__":
    download_model()
