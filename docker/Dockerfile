# Amodelsubmit: Qwen edit + BiRefNet + Hunyuan3D shape + paint
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

LABEL name="amodelsubmit-hunyuan" maintainer="404gen"

# Install docker dependencies
# Installing python 3.11, enbaling channel
RUN apt update -y &&  \
    apt-get install software-properties-common -y &&  \
    apt update -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y &&  \
    apt update -y  &&\
    apt install -y wget gnupg

# Add NVIDIA repo for CUDA 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    lld \
    zstd \
    ninja-build \
    python3.11 \
    python3-pip \
    python3-wheel \
    python3-pybind11 \
    pybind11-dev \
    libjpeg-dev  \
    zlib1g-dev \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# rebinding names for python, pip, gcc, g++
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 120

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;10.0;12.0"

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
RUN pip install -r requirements.txt

# Copy application source
COPY . /app

# Compile C++/CUDA extensions for Hunyuan paint pipeline
# Note: These must be compiled after requirements.txt is installed (torch needed for CUDA extension build)

# 1. Compile mesh_inpaint_processor (pybind11 C++ extension)
WORKDIR /app/modules/hunyuan/paint/DifferentiableRenderer
RUN c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` mesh_inpaint_processor.cpp -o mesh_inpaint_processor`python3-config --extension-suffix` || \
    (echo "Warning: mesh_inpaint_processor compilation failed (has fallback)" && touch mesh_inpaint_processor.so)

# 2. Compile custom_rasterizer (CUDA extension) - REQUIRED for paint pipeline
WORKDIR /app/modules/hunyuan/paint/custom_rasterizer
RUN pip install --no-build-isolation -e . || (echo "ERROR: custom_rasterizer compilation failed - this is required for paint pipeline!" && exit 1)

WORKDIR /app

EXPOSE 10006

# Set entrypoint to use conda environment
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "10006"]
