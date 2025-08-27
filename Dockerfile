FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*


# 한글 설정
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8

WORKDIR /MVTC_Multimodal
COPY . /MVTC_Multimodal

ENV PYTHONPATH=/MVTC_Multimodal

RUN sh setup-dev.sh MVTC
