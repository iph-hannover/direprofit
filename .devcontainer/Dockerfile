# Basisimage mit Python 3.12.7
FROM python:3.12.7

LABEL Name=DireProFit_env Version=0.0.1

# System-Dependencies installieren (einschließlich C++-Tools und CMake)
RUN apt-get update && apt-get install -y \
    qtbase5-dev \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libgl1 \
    libglib2.0-0 \
    libx11-xcb1 \
    git \
    cmake \
    g++ \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt kopieren und installieren
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Installiere scikit-build für eine bessere CMake-Unterstützung
RUN pip install scikit-build