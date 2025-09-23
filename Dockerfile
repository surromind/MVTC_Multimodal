FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Timezone & locale setup
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && echo "Asia/Seoul" > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && locale-gen ko_KR.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8

# System packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        wget \
        curl \
        git \
        gcc \
        g++ \
        make \
        sudo \
        pkg-config \
        libcairo2-dev \
        libgl1 \
        libglib2.0-0 \
        openssh-server \
        awscli \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 installation
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip setup
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install --upgrade pip

# Symlinks for python/pip
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf $(which pip3) /usr/bin/pip \
    && ln -sf $(which pip3) /usr/local/bin/pip

ENV PYTHON_VERSION=3.12 \
    PATH=/usr/local/bin:/usr/bin:$PATH \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/cache/huggingface

WORKDIR /MVTC_Multimodal

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /MVTC_Multimodal/outputs /cache/huggingface

VOLUME ["/MVTC_Multimodal/outputs", "/cache/huggingface"]

ENV PYTHONPATH=/MVTC_Multimodal

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "app/config/default.yaml"]
