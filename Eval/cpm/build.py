import os
import torch
from torch.utils.ffi import create_extension
#from torch.utils.cpp_extension import BuildExtension

sources = ['src/limbs_coco_cpu.c']
headers = ['src/limbs_coco_cpu.h']
defines = []
with_cuda = True

assert torch.cuda.is_available(), "cuda support need"
print('Including CUDA code.')
sources += ['src/nms_cuda.c', 'src/resize_cuda.c']
headers += ['src/nms_cuda.h', 'src/resize_cuda.h']
defines += [('WITH_CUDA', None)]

extra_objects = ['src/nms_layer_kernel.cu.o', 'src/imresize_layer_kernel.cu.o', 'src/limbs_coco.o']

extra_compile_args = ['-fopenmp']

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.cpm',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
)

#ffi = BuildExtension(
#    '_ext.cpm',
#    headers=headers,
#    sources=sources,
#    define_macros=defines,
#    relative_to=__file__,
#    with_cuda=with_cuda,
#    extra_objects=extra_objects,
#    extra_compile_args=extra_compile_args
#)

if __name__ == '__main__':
    ffi.build()
