import os
import sys

from setuptools import setup, find_packages

def iter_protos(parent=None):
    for root, _, files in os.walk('proto'):
        if not files:
            continue
        dest = root if not parent else os.path.join(parent, root)
        yield dest, [os.path.join(root, f) for f in files]

# def get_all_files

pkg_name = 'video-analytics'

setup(name=pkg_name, 
        package_dir={
            '':'lang/python',
            },
        version='0.1.0',
        description='Collection of video analytics for easy integration into python applications',
        author='Nick Burnett',
        author_email='nicholas.c.burnett@gmail.com',
        url='github.com/pvjammer/video-analytics',
        license='Apache License, Version 2.0',
        packages=["object_detectors"],
        data_files=[
            ("darknet-files", ["darknet-files/cfg/yolov3.cfg",
                               "darknet-files/cfg/yolov3-tiny.cfg",
                               "darknet-files/cfg/coco.data",
                               "darknet-files/cfg/coco.names"])
        ],
        install_requires=[
          'cython',
          'opencv-python>=4.2.0.0',
          'numpy>=1.18.0',
          'yolo34py',
          #   'yolo34py-gpu`'
            ],
        py_modules = [
            'object_detectors.__init__'
            ])

