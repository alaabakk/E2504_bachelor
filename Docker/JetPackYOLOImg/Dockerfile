FROM dustynv/l4t-pytorch:r36.4.0

ARG L4T_MAJOR_VERSION=36
ARG L4T_MINOR_VERSION=4
ARG L4T_PATCH_VERSION=3

ARG ZED_SDK_MAJOR=4
ARG ZED_SDK_MINOR=2

#This environment variable is needed to use the streaming features on Jetson inside a container
ENV LOGNAME=root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y || true ; apt-get install --no-install-recommends git lsb-release wget less zstd udev sudo apt-transport-https -y && \
    echo "# R${L4T_MAJOR_VERSION} (release), REVISION: ${L4T_MINOR_VERSION}.${L4T_PATCH_VERSION}" > /etc/nv_tegra_release ; \
    wget -q --no-check-certificate -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/l4t${L4T_MAJOR_VERSION}.${L4T_MINOR_VERSION}/jetsons && \
    chmod +x ZED_SDK_Linux.run ; ./ZED_SDK_Linux.run silent skip_drivers && \
    rm -rf /usr/local/zed/resources/* \
    rm -rf ZED_SDK_Linux.run && \
    rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/lib/aarch64-linux-gnu/tegra/libv4l2.so.0 /usr/lib/aarch64-linux-gnu/libv4l2.so

WORKDIR /usr/local/zed/E2504_bachelor

RUN python3 -m pip install PyOpenGL && \
    python3 -m pip install ultralytics && \
    python3 -m pip install pyserial && \
    python3 -m pip install deep-sort-realtime && \
    git clone https://github.com/alaabakk/E2504_bachelor && \
    cd E2504_bachelor && \
    