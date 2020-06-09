import os
import sys

from setuptools import setup, find_packages

pkg_name = 'video-analytics'

setup(name=pkg_name, 
        package_dir={
            '':'lang/python',
            },
        version='0.1.0',
        description='Collection of easy to usevideo analytics',
        author='Nick Burnett',
        author_email='nicholas.c.burnett@gmail.com',
        url='github.com/pvjammer/video-analytics',
        license='Apache License, Version 2.0',
        packages=["object_detectors"],
        install_requires=[
          'setuptools>=41.0.0',
          'opencv-python>=4.2.0.0'
          'numpy>=1.18.0',
        #   'git+git://github.com/pvjammer/vidstreamer'
            ],
        py_modules = [
            'object_detectors.__init__',
            ])

