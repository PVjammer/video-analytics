from __future__ import print_function, division, unicode_literals, absolute_import

import contextlib
import json
import logging
import os
import select
import sys
import threading
import time

from concurrent import futures

import analytic_pb2
import analytic_pb2_grpc
import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from google.protobuf import json_format

def OptOutVideoLocalization(resp):
    """Opt out of all video localization types for a given video. Modifies resp."""
    resp.opt_out.extend([analytic_pb2.VIDEO_OPT_OUT_TEMPORAL,
                         analytic_pb2.VIDEO_OPT_OUT_SPATIAL])


def OptOutVideoAll(resp):
    """Opt out of all video processing for a given video. Modifies resp."""
    opt_out_video_localization(resp)
    resp.opt_out.append(analytic_pb2.VIDEO_OPT_OUT_DETECTION)


def OptOutVideoLocalization(resp):
    """Opt out of all video localization types for a given video. Modifies resp."""
    resp.opt_out.extend([analytic_pb2.VIDEO_OPT_OUT_TEMPORAL,
                         analytic_pb2.VIDEO_OPT_OUT_SPATIAL])


def OptOutVideoAll(resp):
    """Opt out of all video processing for a given video. Modifies resp."""
    opt_out_video_localization(resp)
    resp.opt_out.append(analytic_pb2.VIDEO_OPT_OUT_DETECTION)


class _AnalyticServicer(analytic_pb2_grpc.AnalyticServicer):
    """The class registered with gRPC, handles endpoints."""

    def __init__(self, svc):
        """Create a servicer using the given Service object as implementation."""
        self.svc = svc

    def DetectImageManipulation(self, req, ctx):
        print("Received request")
        return self.svc._CallEndpoint(self.svc.IMAGE_MANIPULATION, req, analytic_pb2.ImageManipulation(), ctx)

    def DetectVideoManipulation(self, req, ctx):
        return self.svc._CallEndpoint(self.svc.VIDEO_MANIPULATION, req, analytic_pb2.VideoManipulation(), ctx)

    def DetectImageSplice(self, req, ctx):
        return self.svc._CallEndpoint(self.svc.IMAGE_SPLICE, req, analytic_pb2.ImageSplice(), ctx)

    def DetectImageMetaValidity(self, req, ctx):
        return self.svc._CallEndpoint(self.svc.IMAGE_META_VALIDITY, req, analytic_pb2.ImageMetaValidity(), ctx)

class AnalyticService:
    """Actual implementation of the service, with function registration."""

    IMAGE_MANIPULATION = 'ImageManipulation'
    VIDEO_MANIPULATION = 'VideoManipulation'
    IMAGE_SPLICE = "ImageSplice"
    IMAGE_META_VALIDITY = "ImageMetaValidity"

    _ALLOWED_IMPLS = frozenset([IMAGE_MANIPULATION, VIDEO_MANIPULATION, IMAGE_SPLICE, IMAGE_META_VALIDITY])

    def __init__(self):
        self._impls = {}
        self._health_servicer = health.HealthServicer()

    def Start(self, analytic_port=50051, max_workers=10, concurrency_safe=False):
        self.concurrency_safe = concurrency_safe
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                             options=(('grpc.so_reuseport', 0),))
        analytic_pb2_grpc.add_AnalyticServicer_to_server(_AnalyticServicer(self), server)
        health_pb2_grpc.add_HealthServicer_to_server(self._health_servicer, server)
        if not server.add_insecure_port('[::]:{:d}'.format(analytic_port)):
            raise RuntimeError("can't bind to port {}: already in use".format(analytic_port))
        server.start()
        self._health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)
        print("Analytic server started on port {} with PID {}".format(analytic_port, os.getpid()), file=sys.stderr)
        return server

    def Run(self, analytic_port=50051, max_workers=10, concurrency_safe=False):
        server = self.Start(analytic_port=analytic_port, max_workers=max_workers, concurrency_safe=concurrency_safe)

        try:
            while True:
                time.sleep(3600 * 24)
        except KeyboardInterrupt:
            server.stop(0)
            logging.info("Server stopped")
            return 0
        except Exception as e:
            server.stop(0)
            logging.error("Caught exception: %s", e)
            return -1

    def RegisterImageManipulation(self, f):
        return self._RegisterImpl(self.IMAGE_MANIPULATION, f)

    def RegisterVideoManipulation(self, f):
        return self._RegisterImpl(self.VIDEO_MANIPULATION, f)

    def RegisterImageSplice(self, f):
        return self._RegisterImpl(self.IMAGE_SPLICE, f)

    def RegisterImageMetaValidity(self, f):
        return self._RegisterImpl(self.IMAGE_META_VALIDITY, f)

    def _RegisterImpl(self, type_name, f):
        if type_name not in self._ALLOWED_IMPLS:
            raise ValueError("unknown implementation type {} specified".format(type_name))
        if type_name in self._impls:
            raise ValueError("implementation for {} already present".format(type_name))
        self._impls[type_name] = f
        return self

    def _CallEndpoint(self, ep_type, req, resp, ctx):
        """Implements calling endpoints and handling various exceptions that can come back.

        Args:
            ep_type: The name of the manipulation, e.g., "image". Should be in ALLOWED_IMPLS.
            req: The request proto to send.
            resp: The response proto to fill in.
            ctx: The context, used mainly for aborting with error codes.

        Returns:
            An appropriate response object for the endpoint type specified.
        """
        ep_func = self._impls.get(ep_type)
        print("EPFUNC: {!s}".format(ep_func))
        if not ep_func:
            ctx.abort(grpc.StatusCode.UNIMPLEMENTED, "Endpoint {!r} not implemented".format(ep_type))

        try:
            print("Calling function for: {!s}".format(ep_type))
            ep_func(req, resp)
        except ValueError as e:
            logging.exception('invalid input')
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "Endpoint {!r} invalid input: {}".format(ep_type, e))
        except NotImplementedError as e:
            logging.warn('unimplemented endpoint {}'.format(ep_type))
            ctx.abort(grpc.StatusCode.UNIMPLEMENTED, "Endpoint {!r} not implemented: {}".format(ep_type, e))
        except Exception as e:
            logging.exception('unknown error')
            ctx.abort(grpc.StatusCode.UNKNOWN, "Error processing endpoint {!r}: {}".format(ep_type, e))
        return resp


class FIFOTimeoutError(IOError):
    def __init__(self, op, timeout):
        return super(FIFOTimeoutError, self).__init__("timed out with op {!r} after {} seconds".format(op, timeout))


class FIFOContextAbortedError(IOError):
    def __init__(self, code, details):
        self.code = code
        self.details = details
        super(FIFOContextAbortedError, self).__init__("Context aborted with code: {!s}.  Message: {!s}".format(code, details))


class FIFOContext:
    def abort(self, code, details):
        raise FIFOContextAbortedError(code, details)


c
