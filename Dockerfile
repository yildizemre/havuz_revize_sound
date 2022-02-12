FROM tensorflow/tensorflow:2.3.1-gpu

ARG DEBIAN_FRONTEND=noninteractive 


RUN add-apt-repository ppa:apt-fast/stable -y 
RUN apt-get update 

RUN apt-get install apt-fast -y --no-install-recommends 
RUN apt-fast upgrade -y 
RUN apt-fast install -yq --no-install-recommends git curl

#libx264
RUN apt-fast install libx264-dev -yq --no-install-recommends

#libx265
RUN apt-fast install libx264-dev -yq --no-install-recommends && \ 
    apt-fast install libx265-dev -yq --no-install-recommends && \ 
    apt-fast install libnuma-dev -yq --no-install-recommends

#libvpx
RUN  apt-fast install libvpx-dev -yq --no-install-recommends

#libfdk-aac
RUN  apt-fast install libfdk-aac-dev -yq --no-install-recommends

#libmp3lame
RUN  apt-fast install libmp3lame-dev -yq --no-install-recommends
#libopus
RUN apt-fast install libopus-dev -yq --no-install-recommends


RUN apt-fast install -yq libunistring-dev cuda-npp-10-1 \
    cuda-npp-dev-10-1 --no-install-recommends

RUN apt-get update && apt-get install apt-utils -y
RUN apt-get update && apt-get install -y \
    bc \
    build-essential \
    cmake \
    curl \
    g++ \
    gfortran \
    git \
    libffi-dev \
    libfreetype6-dev \
    libhdf5-dev \
    libjpeg-dev \
    liblcms2-dev \
    libopenblas-dev \
    liblapack-dev \
    libopenjp2* \
    libpng-dev \
    libssl-dev \
    libtiff5-dev \
    libwebp-dev \
    libzmq3-dev \
    nano \
    pkg-config \
    python-dev \
    software-properties-common \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    qt5-default \
    libvtk6-dev \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libgdal-dev \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev \
    libtbb-dev \
    libeigen3-dev \
    python-dev \
    python-tk \
    python-numpy \
    python3-dev \
    python3-tk \
    python3-numpy \
    ant \
    default-jdk \
    doxygen


# OpenCV dependency

RUN echo "deb http://security.ubuntu.com/ubuntu xenial-security main" | tee -a /etc/apt/sources.list && \
    apt update -y

RUN  apt-fast update && \ 
    apt-fast install cmake g++ python3-dev \
    python3-numpy \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libtbb2 \
    libtbb-dev \
    libjasper1 \
    libjasper-dev \
    libpq-dev \
    libhdf5-dev -y && \
    apt-fast autoremove 


RUN python3 -m pip install --upgrade pip

# Add SNI support to Python
RUN pip3 --no-cache-dir install \
    pyopenssl \
    ndg-httpsclient \
    pyasn1 \
    setuptools \
    future>=0.17.1 

ENV OPENCV_VERSION=4.5.0
ENV OPENCV_PYTHON_VERSION=9.2

RUN mkdir /temp \ 
    && wget -nv https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O /temp/opencvcontrib-${OPENCV_VERSION}.zip \
    && wget -nv https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O /temp/opencv-${OPENCV_VERSION}.zip \
    && wget -nv https://github.com/VLAM3D/opencv-python/archive/${OPENCV_PYTHON_VERSION}.zip -O /temp/opencv-python.zip \
    && unzip /temp/opencv-${OPENCV_VERSION}.zip\
    && unzip /temp/opencvcontrib-${OPENCV_VERSION}.zip\
    && unzip /temp/opencv-python.zip

RUN apt-fast install -yq libcudnn7-dev libopenjp2-7-dev libopenjp2-7 libopenjpip7 libopenjpip-dec-server libtiff5-dev libtiff5 libopenjpip-server
#RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so.10.2.2.214 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so
#RUN apt-fast install -yq qt5-default

RUN apt-fast -yq install cuda-cufft-dev-10-1
RUN apt-fast -yq install libavfilter-dev
#RUN apt-fast -yq install libcublas-dev cuda-cublas-dev-10-0 cuda-cublas-10-0 libcublas-*
RUN apt-fast -yq install cuda-libraries-dev-10-1

RUN echo 'PATH="/usr/local/cuda-10.1/targets/x86_64-linux/lib:$PATH"' >> ~/.bashrc
RUN echo 'PATH="/usr/local/cuda-10.1/targets/x86_64-linux/include:$PATH"' >> ~/.bashrc
RUN echo 'CPATH=/usr/local/cuda-10.1/targets/x86_64-linux/include:$CPATH' >> ~/.bashrc
RUN echo 'LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
RUN echo 'LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib:$LIBRARY_PATH' >> ~/.bashrc
RUN echo 'PATH=/usr/local/cuda-10.1/bin:$PATH' >> ~/.bashrc

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcublas-dev_10.1.0.105-1_amd64.deb && \
    dpkg -i libcublas-dev_10.1.0.105-1_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcublas10_10.1.0.105-1_amd64.deb && \
    dpkg -i libcublas10_10.1.0.105-1_amd64.deb && \
    cp /usr/include/cublas* /usr/local/cuda-10.1/include/ && \
    cp /usr/include/cublas* /usr/local/cuda-10.1/targets/x86_64-linux/lib && \
    cp /usr/lib/x86_64-linux-gnu/libcublas* /usr/local/cuda-10.1/targets/x86_64-linux/lib && \
    cp /usr/lib/x86_64-linux-gnu/libcublas* /usr/local/cuda-10.1/targets/x86_64-linux/lib

RUN source ~/.bashrc && cd /opencv-${OPENCV_VERSION} && mkdir build && cd build && \
    # BUILD SHARED LIBS FOR C++ DEV WITH CUDA
    cmake .. -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DCUDA_cufft_LIBRARY=/usr/local/cuda-10.1/targets/x86_64-linux/lib \
    -DCUDA_cublas_LIBRARY=/usr/local/cuda-10.1/targets/x86_64-linux/lib \
    -DCUDA_CUBLAS_LIBRARIES=/usr/local/cuda-10.1/targets/x86_64-linux/lib \
    -DCUDA_CUFFT_LIBRARIES=/usr/local/cuda-10.1/targets/x86_64-linux/lib \
    -DCUDA_INCLUDE_DIRS=/usr/local/cuda-10.1/targets/x86_64-linux/include \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
    -DCUDA_BIN_PATH=/usr/local/cuda-10.1 \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DOPENCV_DNN_CUDA=ON \
    -DCUDA_ARCH_BIN=7.5 \
    -DCUDA_ARCH_PTX=7.5 \
    -DWITH_CUBLAS=ON \
    -DOpenGL_GL_PREFERENCE=GLVND \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_MKL=ON \
    -DWITH_QT=5 \
    -DMKL_WITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DBUILD_PROTOBUF=ON \
    -DWITH_V4L=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMER=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_NEW_PYTHON_SUPPORT=ON \
    -DBUILD_opencv_python3=ON \
    -DHAVE_opencv_python3=ON \
    -DPROTOBUF_INCLUDE_DIR=/protobuf \
    -DPROTOBUF_LIBRARY=/usr/local/lib \
    -DPYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
    -DBUILD_opencv_legacy=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local


RUN source ~/.bashrc && cd /opencv-${OPENCV_VERSION}/build && make -j $(nproc) install 


RUN apt --fix-broken install -y

# Install dependencies for Caffe
RUN apt-fast update && apt-fast install \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev -y \
    protobuf-compiler && \
    apt-fast clean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install Caffe
RUN apt-get update && apt-get install caffe-cuda -y


ARG THEANO_VERSION=rel-0.8.2

# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip3 --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
    \
    echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
    \n[lib]\ncnmem=0.95 \
    \n[nvcc]\nfastmath=True \
    \n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
    \n[DebugMode]\ncheck_finite=1" \
    > /root/.theanorc


#Dlib
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake .. -DLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 .. \
    && cmake --build . --config Release \
    && cd ../ \
    && python3 setup.py install 



#Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
#especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-fast update && apt-fast install -yq --no-install-recommends \
    python3-numpy \
    python3-scipy \
    python3-nose \
    python3-h5py \
    python3-skimage \
    python3-matplotlib \
    python3-pandas \
    python3-sklearn \
    python3-sympy &&\
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install py-zipkin && pip3 install pika && pip3 install python-dotenv

RUN pip3 install -U scipy psutil scikit-image

WORKDIR /usr/src/app 

COPY ./  /usr/src/app
COPY requirements.txt ./ 

RUN pip3 install --no-cache-dir -r requirements.txt 

RUN apt-get update
RUN apt-get install net-tools openssh-server iptables iputils-ping -y

# RUN pip3 --no-cache-dir install pika python-dotenv redis pybase64 uuid graypy

CMD ["python3","/usr/src/app/time_object_tracker.py"]

# docker run  --gpus all -d --restart=always  --name pool-gpu  pool
