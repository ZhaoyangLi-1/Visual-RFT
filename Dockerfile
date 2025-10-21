# -------- Base image: CUDA 11.8 + cuDNN + Ubuntu 22.04 --------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# -------- System deps --------
RUN apt-get update && apt-get install -y \
    wget git ffmpeg libgl1-mesa-glx libglfw3 patchelf \
    python3-opencv libsm6 libxext6 libxrender-dev cmake unzip bzip2 \
    tmux zip vim libglew-dev libgl1 libglvnd-dev libegl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------- Miniconda --------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && bash ~/miniconda.sh -b -p $CONDA_DIR \
 && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# -------- 关键：改用 conda-forge，移除 defaults，避免 TOS --------
RUN conda config --system --remove-key default_channels || true && \
    conda config --system --add default_channels https://conda.anaconda.org/conda-forge && \
    conda config --system --remove channels defaults || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict

# 装 mamba（现在只走 conda-forge，不会触发 TOS）
RUN conda install -n base -y mamba && conda clean -afy

# 建环境
RUN mamba create -n vrft -y python=3.10 && conda clean -afy
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=vrft
ENV PATH=$CONDA_DIR/envs/vrft/bin:$PATH

# -------- Clone repo --------
WORKDIR /root
RUN git clone https://github.com/ZhaoyangLi-1/Visual-RFT.git

# -------- Python deps --------
WORKDIR /root/Visual-RFT
RUN bash setup.sh

# -------- Default command --------
CMD ["bash"]
