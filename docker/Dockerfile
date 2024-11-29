FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y \
    wget \
    htop \
    git \
    nano \
    cmake \
    unzip \
    zip \
    vim \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libopencv-dev \
    libglew-dev \
    assimp-utils libassimp-dev \
    libboost-all-dev \
    libglfw3-dev \
    libgtk-3-dev \
    ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
    libeigen3-dev \
    libgl1-mesa-dev xorg-dev \
    libembree-dev
    
RUN ln -s /lib/x86_64-linux-gnu/libembree3.so /lib/x86_64-linux-gnu/libembree.so
    
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV PYOPENGL_PLATFORM egl

#RUN ls /usr/share/glvnd/egl_vendor.d/
#COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# fixuid
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# miniconda
WORKDIR /home/$USERNAME
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/$USERNAME/miniconda/bin:$PATH

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda
    # && \
    # rm ~/miniconda.sh && \


#RUN echo 112
COPY docker/environment.yml /home/$USERNAME/environment.yml
RUN conda env create -f /home/$USERNAME/environment.yml
ENV PATH=/home/$USERNAME/miniconda/envs/bbsplat/bin:$PATH

RUN echo "source activate bbsplat" > ~/.bashrc
ENV PATH /opt/conda/envs/bbsplat/bin:$PATH

# python libs
RUN pip install --upgrade pip


# docker setup
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
