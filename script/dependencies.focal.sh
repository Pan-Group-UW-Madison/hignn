#!/bin/sh

apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    clang-format \
    cmake \
    g++-9 \
    gcc-9 \
    gcc-9-base \
    gcc-9-plugin-dev \
    git \
    openssh-client \
    openssl \
    python3.9 \
    python3.9-dev \
    python3-pip \
    unzip \
    valgrind \
    wget \
    libtbb-dev \
    libnuma-dev \
    && update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 \
        --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-9 \
        --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
        --slave /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-9 \
        --slave /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-9 \
        --slave /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcc-ar x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcc-nm x86_64-linux-gnu-gcc-nm /usr/bin/x86_64-linux-gnu-gcc-nm-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcc-ranlib x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcov x86_64-linux-gnu-gcov /usr/bin/x86_64-linux-gnu-gcov-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcov-dump x86_64-linux-gnu-gcov-dump /usr/bin/x86_64-linux-gnu-gcov-dump-9 \
        --slave /usr/bin/x86_64-linux-gnu-gcov-tool x86_64-linux-gnu-gcov-tool /usr/bin/x86_64-linux-gnu-gcov-tool-9 \
    || exit 1

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1