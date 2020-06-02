FROM datamachines/cudnn_tensorflow_opencv:10.2_1.15.2_4.2.0-20200211

RUN apt install -y curl \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf \
    && ldconfig

RUN mkdir /wrk \
    && cd /wrk \
    && git clone https://github.com/pjreddie/darknet \
    && cd darknet \
    && git config --global user.email "help@datamachines.io" \
    && git config --global user.name "dmc" \
    && git checkout 61c9d02ec461e30d55762ec7669d6a1d3c356fb2 \
    && curl -L https://github.com/pjreddie/darknet/pull/1348.patch > 1348.patch \
    && git am 1348.patch \
    && sed -i 's/GPU=0/GPU=1/;s/CUDNN=0/CUDNN=1/;s/OPENCV=0/OPENCV=1/;s/OPENMP=0/OPENMP=1/;s%-L/usr/local/cuda/lib64%-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs%' Makefile \
    && make

ENV DARKNET_HOME /wrk/darknet
ENV LD_LIBRARY_PATH /wrk/darknet

## Download and compile YOLO3-4-Py ##
ENV GPU 1
# Not compiling with OpenCV [need 3.4, we are based on 4.x] + we are only using the 'detect' python function, OpenCV4 is functional on host and in Darknet already
#ENV OPENCV 1
ENV OPENMP 1
WORKDIR /
RUN git clone https://github.com/madhawav/YOLO3-4-Py.git
WORKDIR /YOLO3-4-Py
RUN pip3 install pkgconfig
RUN pip3 install cython
RUN python3 setup.py build_ext --inplace
RUN python3 setup.py install

RUN pip install git+git://github.com/PVjammer/vidstreamer.git





