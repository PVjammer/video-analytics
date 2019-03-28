import os
import sys

def iter_protos(parent=None):
    for root, _, files in os.walk('proto'):
        if not files:
            continue
        dest = root if not parent else os.path.join(parent, root)
        yield dest, [os.path.join(root, f) for f in files]

from setuptools import setup, find_packages

pkg_name = 'ace'

setup(name=pkg_name, 
        pkg_dir={
            '':'lang/python',
            },
        version='1.0.0',
        description='Protocol wrapper for streaming video analytics',
        author='Nick Burnett, Data Machines Corp.',
        author_email='nicholasburnett@datamachines.io',
        url='github.com/PVjammer/video-analytics/analyticframework/lang/py',
        packages=find_packages(),
        install_requires=[
            'setuptools==39.0.1',
            'grpcio==1.15.0',
            'grpcio-tools==1.15.0',
            'grpcio_health_checking==1.15.0',
            'protobuf==3.6.1',
            ],
        data_files=list(iter_protos("analyticframework/proto")),
        py_modeules = [
            'analytic_pb2',
            'analytic_pb2._grpc',
            'analyticservice',
            'google/rpc/__init__',
            'google/rpc/status_pb2',
            'google/rpc/code_pb2',
            'google/rpc/error_details_pb2'
            ])


