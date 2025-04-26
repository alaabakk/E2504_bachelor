FROM dustynv/l4t-pytorch:r36.4.0

ARG L4T_MAJOR_VERSION=36
ARG L4T_MINOR_VERSION=4
ARG L4T_PATCH_VERSION=3

ARG ZED_SDK_MAJOR=4
ARG ZED_SDK_MINOR=2

ENV LOGNAME=root
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages + Git LFS
RUN apt-get update -y || true && \
    apt-get install --no-install-recommends -y \
        git git-lfs lsb-release wget less zstd udev sudo apt-transport-https && \
    rm -rf /var/lib/apt/lists/*

# Install ZED SDK
RUN echo "# R${L4T_MAJOR_VERSION} (release), REVISION: ${L4T_MINOR_VERSION}.${L4T_PATCH_VERSION}" > /etc/nv_tegra_release && \
    wget -q --no-check-certificate -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/l4t${L4T_MAJOR_VERSION}.${L4T_MINOR_VERSION}/jetsons && \
    chmod +x ZED_SDK_Linux.run && \
    ./ZED_SDK_Linux.run silent skip_drivers && \
    rm -rf /usr/local/zed/resources/* && \
    rm -f ZED_SDK_Linux.run

# Fix V4L2 library link
RUN ln -sf /usr/lib/aarch64-linux-gnu/tegra/libv4l2.so.0 /usr/lib/aarch64-linux-gnu/libv4l2.so

# Install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        PyOpenGL \
        ultralytics \
        pyserial \
        deep-sort-realtime \
        openpyxl

# Clone your repo
WORKDIR /usr/local/zed
RUN git clone https://github.com/alaabakk/E2504_bachelor.git && \
    cd E2504_bachelor && \
    git lfs install && \
    git lfs pull

# Set working directory
WORKDIR /usr/local/zed/E2504_bachelor

<<<<<<< HEAD
=======
RUN python3 -m pip install PyOpenGL && \
    python3 -m pip install ultralytics && \
    python3 -m pip install pyserial && \
    python3 -m pip install deep-sort-realtime && \
    git clone https://github.com/alaabakk/E2504_bachelor && \
    cd E2504_bachelor && \
    
>>>>>>> c791d0392d9412c9b84fb202ba8907ea4e78a761
