#!/bin/bash

root_dir=$(pwd)

build_dir=$root_dir/opencv-4.1.0/build
install_dir=$root_dir/install
rm -rf $build_dir
if [ ! -d $build_dir ]; then
    mkdir $build_dir
fi
cd $build_dir

cmake -DPYTHON_EXECUTABLE=/usr/local/bin/python -DBUILD_opencv_python_bindings_generator=OFF -DOPENCV_GENERATE_PKGCONFIG=YES -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_OPENEXR=OFF -DWITH_PYTHON=OFF -DBUILD_FAT_JAVA_LIB=OFF -DBUILD_JAVA=OFF -DWITH_GSTREAMER=OFF -DWITH_FFMPEG=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF -DWITH_IPP=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DWITH_INF_ENGINE=OFF -DENABLE_CXX11=ON -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_TIFF=ON -DBUILD_PNG=ON -DCMAKE_INSTALL_PREFIX=$install_dir ..

make -j16
make install

cd $root_dir

