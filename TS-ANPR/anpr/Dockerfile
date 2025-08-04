# Raspberry Pi (aarch64)용 Python3.11 기반 이미지
FROM python:3.11-slim-bullseye

# 시스템 패키지 업데이트 및 의존성 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    ffmpeg \
    libfontconfig1 \
    fonts-dejavu \
    curl \
    git \
    usbutils \
    build-essential \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libcap-dev \
    libopenblas-dev \
    libavformat-dev \
    libswscale-dev \
    libavcodec-dev \
    libavdevice-dev \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# 파이썬 파일 복사
COPY . /app

# 파이썬 의존성 설치
RUN pip install --upgrade pip && \
    pip install opencv-python numpy Pillow requests pyserial

# 실행 엔트리포인트
CMD ["python", "main.py"]
