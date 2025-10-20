# -------- Base image: CUDA 12.8 + cuDNN9 + Ubuntu 22.04 --------
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# -------- Install dependencies --------
RUN apt-get update && apt-get install -y \
    wget git ffmpeg libgl1-mesa-glx libglfw3 libosmesa6-dev patchelf \
    python3-opencv libsm6 libxext6 libxrender-dev cmake unzip bzip2 \
    tmux zip vim libglew-dev libgl1 libglvnd-dev libegl1 \
    && rm -rf /var/lib/apt/lists/*

# -------- Install Miniconda --------
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# -------- Install mamba --------
RUN conda install -n base -c conda-forge mamba && conda clean -afy

# -------- Copy conda environment --------
COPY conda_environment.yaml /environment.yml

RUN mamba env create -v -f /environment.yml python=3.10 && conda clean -afy

# -------- Install MuJoCo 2.1.0 --------
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    mkdir -p /root/.mujoco && \
    tar -xvf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/ && \
    rm mujoco210-linux-x86_64.tar.gz

# -------- Copy MuJoCo license key --------
COPY mjkey.txt /root/.mujoco/mjkey.txt

# -------- Set environment variables for MuJoCo --------
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
ENV PATH=$LD_LIBRARY_PATH:$PATH
ENV MUJOCO_GL=egl 

# -------- Set shell to bash --------
SHELL ["/bin/bash", "-c"]

# -------- Clone repo --------
WORKDIR /root
RUN git clone https://github.com/ZhaoyangLi-1/Visual-RFT.git

# -------- Set working directory --------
WORKDIR /root/Visual-RFT
RUN pip uninstall -y transformers huggingface-hub
RUN pip install "transformers>=4.51.0,<5.0" "huggingface-hub>=0.23,<1.0"
RUN bash setup.sh

# -------- Default command --------
CMD ["bash"]
