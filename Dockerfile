FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Timezone & locale
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    apt-get update && \
    apt-get install -y locales && \
    locale-gen ko_KR.UTF-8

ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        wget curl git gcc g++ make sudo \
        openssh-server awscli \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 설치
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --upgrade pip

# python, pip 심볼릭 링크
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf $(which pip3) /usr/bin/pip && \
    ln -sf $(which pip3) /usr/local/bin/pip

# 환경변수 (경로는 디렉토리 기준으로 수정)
ENV PYTHON_VERSION=3.12 \
    PATH=/usr/local/bin:/usr/bin:$PATH \
    PYTHONIOENCODING=utf-8 \
    LC_ALL=C.UTF-8

RUN python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /MVTC_Multimodal
COPY . /MVTC_Multimodal

ENV PYTHONPATH=/MVTC_Multimodal

