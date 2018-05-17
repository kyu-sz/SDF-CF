#!/usr/bin/env bash
# Install dependencies.
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip \
      python-yaml
sudo pip install \
      future \
      numpy \
      protobuf \
      typing

# Build the project.
mkdir build
cd build
cmake ..
make -j 8
sudo make install