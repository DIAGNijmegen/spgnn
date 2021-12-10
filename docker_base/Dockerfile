ARG UBUNTU_VERSION=18.04
ARG CUDA_MAJOR_VERSION=11.1
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.6.5
ARG NUMPY_VERSION=1.19.5
ARG PYTORCH_VERSION=1.9.0
ARG TORCHVISION_VERSION=0.10.0

ARG TENSORFLOW_VERSION=2.4.0

ARG BUILD_JOBS=32



FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# === Propagate build args ===
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG BUILD_JOBS

# === Install build packages ===
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libatlas-base-dev \
        libblas-dev \
        libopenblas-dev \
        liblapack-dev \
        build-essential \
        pkg-config apt-transport-https \
        openjdk-8-jdk \
        g++ ninja-build make \
        wget git zip unzip \
        libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        gcovr libffi-dev uuid-dev


# === Install python ===
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd /tmp && \
    rm -rf Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade pipenv && \
    pip3 install cython wheel numpy==${NUMPY_VERSION}

# === Install cmake ===cmake-3.20.0-rc3-linux-x86_64.tar.gz
RUN wget https://cmake.org/files/v3.20/cmake-3.20.0-rc3-linux-x86_64.tar.gz -O /tmp/cmake.tgz && \
    tar xf /tmp/cmake.tgz -C /opt && \
    rm /tmp/cmake.tgz && \
    mv /opt/cmake* /opt/cmake

RUN apt-get update && apt-get install -y \
  python-pydot-ng \
  python-pydot \
  graphviz \
  python3-tk 
ENV PATH "${PATH}:/opt/cmake/bin"

FROM builder as numpy

# === Propagate build args ===
ARG NUMPY_VERSION

# === Download numpy source code and compile ===
RUN wget "https://github.com/numpy/numpy/releases/download/v${NUMPY_VERSION}/numpy-${NUMPY_VERSION}.tar.gz" -O /tmp/numpy.tgz && \
    mkdir /numpy && \
    tar xf /tmp/numpy.tgz -C /numpy && \
    rm /tmp/numpy.tgz && \
    mv /numpy/numpy-* /numpy/src && \
    cd /numpy/src && \
    python3 setup.py build_ext --inplace

# === Copy include folder to different location ===
RUN NUMPY_VERSION_MAJOR_MINOR=`python3 -c "import re; print(re.match(r'[0-9]+\.[0-9]+', '${NUMPY_VERSION}').group())"` && \
    mkdir -p "/numpy/numpy-$NUMPY_VERSION_MAJOR_MINOR" && \
    cp -r /numpy/src/numpy/core/include /numpy/numpy-$NUMPY_VERSION_MAJOR_MINOR/include


  
FROM builder AS apex

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION

# === Install PyTorch ===
RUN CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g"` && \
    pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        "torch==${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}"

# === Build the library ===
ENV TORCH_CUDA_ARCH_LIST "5.0+PTX;6.0;6.1;7.0;7.5;8.0"
RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    python3 setup.py --cpp_ext --cuda_ext bdist_wheel && \
    mkdir /apex && \
    cp dist/*.whl /apex/


FROM builder AS dgl

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION


# === Install PyTorch ===
RUN CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g"` && \
    pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        "torch==${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}"

# === Build the library ===
ENV TORCH_CUDA_ARCH_LIST "5.0+PTX;6.0;6.1;7.0;7.5;8.0"
RUN git clone --recurse-submodules https://github.com/dmlc/dgl.git /tmp/dgl && \
    cd /tmp/dgl && \
    mkdir /build && cmake -DUSE_CUDA=ON && \
    make -j4
ENV DGL_LIBRARY_PATH=/tmp/dgl
RUN cd /tmp/dgl/python && python3 setup.py bdist_wheel && \
    mkdir /dgl && \
    cp dist/*.whl /dgl/


FROM builder AS base

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TENSORFLOW_VERSION
ARG BUILD_JOBS

# === Configurate environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# === Install libs ===
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        man \
        openssh-server \
        ca-certificates apt-transport-https \
        sudo \
        git subversion \
        nano vim \
        tmux screen \
        htop \
        g++ ninja-build \
        rsync \
        pv \
        curl wget \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    cat /etc/sudoers | grep -v secure_path > /tmp/sudoers && mv /tmp/sudoers /etc/sudoers

RUN env | grep '^PATH=\|^LD_LIBRARY_PATH=\|^LANG=\|^LC_ALL=\|^CUDA_ROOT=' > /etc/environment

# === Configure timezone ===
RUN echo "Europe/Amsterdam" > /etc/timezone && \
    rm -f /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# === Setup user ===
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user ssh && \
    gpasswd -a user sudo


# === Install python with up-to-date pip ===
RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd ~ && \
    rm -rf /tmp/Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip && \
    rm -rf ~/.cache/pip

# === Install python libraries ===
COPY install_files/requirements.in /root/python-packages/
COPY --from=apex /apex/apex-*-linux_x86_64.whl /root/python-packages/
COPY --from=dgl /dgl/dgl-*-linux_x86_64.whl /root/python-packages/

RUN cd /root/python-packages && \
    CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g"` && \
    sed -i \
        -e "s|%PYTORCH_VERSION%|${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        -e "s|%TORCHVISION_VERSION%|${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        -e "s|%APEX_WHEEL%|`find /root/python-packages -name 'apex-*.whl'`|g" \
        -e "s|%DGL_WHEEL%|`find /root/python-packages -name 'dgl-*.whl'`|g" \
        requirements.in && \
    pip-compile requirements.in --verbose --find-links https://download.pytorch.org/whl/torch_stable.html && \
    pip-sync && \
    rm -rf ~/.cache/pip*

# === Set some environment variables for TensorFlow. ===
ENV FOR_DISABLE_CONSOLE_CTRL_HANDLER 1
ENV TF_CPP_MIN_LOG_LEVEL 3
RUN env | grep '^FOR_DISABLE_CONSOLE_CTRL_HANDLER=\|^TF_CPP_MIN_LOG_LEVEL=' >> /etc/environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive


RUN ldconfig

# === Configure stop signal. ===
STOPSIGNAL SIGINT

## === Expose ports for ssh, tensorboard, and jupyter. ===
#EXPOSE 22 6006 8888
#
#
#FROM base AS FB
#
#RUN apt-get update && \
#    apt-get install -y swig3.0
#
#RUN cd /tmp && \
#    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
#    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#
#RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
#
#RUN apt-get update && \
#    apt-get install -y intel-mkl-64bit-2019.3-062
#
#
### update alternatives
#RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
#RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
#RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
#RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
#
#RUN echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf
#RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
#RUN ldconfig
#
#RUN echo "MKL_THREADING_LAYER=GNU" >> /etc/environment


## === Build the library ===
#RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
#    cd /tmp/faiss && \
#    cmake -B build . -DCUDAToolkit_ROOT=/usr/local/cuda-11.0 -DCMAKE_CUDA_ARCHITECTURES="80" -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON && \
#    make -C build -j8 faiss && make -C build -j8 swigfaiss && make -C build install && cd build/faiss/python && python3 setup.py build
#
#RUN cd /tmp/faiss && \
#    PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
#
## === Build the library ===
#RUN git clone https://github.com/kyamagu/faiss-wheels.git /tmp/faissw && \
#    cd /tmp/faissw  && \
#    cmake -B build . -DCUDAToolkit_ROOT=/usr/local/cuda-11.0 -DCMAKE_CUDA_ARCHITECTURES="80" -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON && \
#    make -C build -j8 faiss && make -C build -j8 swigfaiss && make -C build install && cd build/faiss/python && python3 setup.py build

 
#RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
#    cd /tmp/faiss && \
#    ./configure --prefix=/usr --libdir=/usr/local/cuda-11.0  && \
#    make -j8 && make -C python && make install


#ENV PYTHONPATH="/usr/src/app/faiss-1.3.0/python/:${PYTHONPATH}"

#RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
#    cd /tmp/faiss && \
#    ./configure &&\
#    make &&\
#    make install &&\
#    cd gpu &&\
#    make -j &&\
#    cd ../python &&\
#    make _swigfaiss_gpu.so &&\
#    cd ../ &&\
#    make py &&\

# === Configure stop signal. ===
STOPSIGNAL SIGINT

# === Expose ports for ssh, tensorboard, and jupyter. ===
EXPOSE 22 6006 8888
USER root
#WORKDIR /share/airway_labeling_t7828/bronchial
COPY run_files/run.sh /tmp/run.sh
ENTRYPOINT ["/bin/bash", "/tmp/run.sh"]


