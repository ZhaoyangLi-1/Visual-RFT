# -------- Base image: CUDA 12.8 + cuDNN + Ubuntu 22.04 --------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# -------- System deps --------
RUN apt-get update && apt-get install -y \
    wget git ffmpeg libgl1-mesa-glx libglfw3 patchelf \
    python3-opencv libsm6 libxext6 libxrender-dev cmake unzip bzip2 \
    tmux zip vim libglew-dev libgl1 libglvnd-dev libegl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------- Miniconda + mamba --------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && bash ~/miniconda.sh -b -p $CONDA_DIR \
 && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -n base -c conda-forge mamba && conda clean -afy

RUN mamba create -n vrft -y python=3.10 && conda clean -afy
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=vrft
ENV PATH=$CONDA_DIR/envs/vrft/bin:$PATH

# -------- Clone repo --------
WORKDIR /root
RUN git clone https://github.com/ZhaoyangLi-1/Visual-RFT.git

# -------- Python deps --------
WORKDIR /root/Visual-RFT
RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip uninstall -y transformers huggingface-hub || true \
 && python -m pip install --no-cache-dir "transformers>=4.51.0,<5.0" "huggingface-hub>=0.23,<1.0"

# 项目安装
RUN bash setup.sh

# -------- Default command --------
CMD ["bash"]
