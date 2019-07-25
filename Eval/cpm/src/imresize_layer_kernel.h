#ifndef _IMRESIZE_KERNEL
#define _IMRESIZE_KERNEL

#include "math_functions.h"

#ifdef __cplusplus
extern "C" {
#endif


void imresize_ongpu(const Dtype * src_ptr, Dtype *dst_ptr, const int num, const int channel, const int height, const int width,
                const int targetSpatialHeight, const int targetSpatialWidth, const float start_scale, const float scale_gap, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif
