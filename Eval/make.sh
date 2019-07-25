#!/usr/bin/env bash

NVCC=/usr/local/cuda/bin/nvcc


cd cpm/src
echo "Compiling im_resize layer kernels by $NVCC..."
$NVCC -c -o imresize_layer_kernel.cu.o imresize_layer_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

echo "Compiling nms layer kernels by $NVCC..."
$NVCC -c -o nms_layer_kernel.cu.o nms_layer_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

echo "Compiling limbs_coco.cpp..."
gcc -c -o limbs_coco.o limbs_coco.cpp -fPIC -std=c++11

cd ../
python3 build.py
