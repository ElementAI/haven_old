FROM ubuntu:16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.1.85

ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.1"


RUN apt-get update --fix-missing
RUN apt-get update -y
RUN apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion vim emacs


#RUN apt-get update && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2 \
         libnccl-dev \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh


ENV PATH="/opt/conda/bin:${PATH}"

RUN /opt/conda/bin/pip install \
    matplotlib \
    plotly \
    dominate \
    opencv-python \
    tqdm \
    pyamg

RUN conda install scipy scikit-image scikit-learn

RUN pip install beautifulsoup4



RUN pip install lxml

RUN conda install ffmpeg 

RUN conda install -c numba numba 

RUN pip install h5py
RUN pip install cython
RUN pip install pycocotools

RUN pip install chainercv
RUN pip install pandas
RUN pip install tqdm
RUN pip install bs4
RUN pip install \
    matplotlib \
    plotly \
    dominate \
    opencv-python \
    tqdm \
    pyamg
RUN pip install lxml

RUN apt-get update \
    && apt-get install -y \
        nmap \
        vim
RUN pip install ipdb
RUN pip install pyyaml

RUN pip install torchgeometry

RUN pip install pyzmq

RUN pip install torchfile
RUN pip install websocket-client

RUN apt-get install gcc-5
RUN pip install gputil
RUN apt-get update
RUN apt-get install -y vim-tiny


RUN conda install pytorch torchvision cuda90 -c pytorch
RUN apt-get -y install tmux

RUN pip install structlog
RUN pip install baal
RUN pip install umap-learn
RUN pip install notebook

CMD ["/bin/bash"]

