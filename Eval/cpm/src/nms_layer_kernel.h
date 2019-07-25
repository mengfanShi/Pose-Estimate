#ifndef _NMS_KERNEL
#define _NMS_KERNEL

#include "math_functions.h"

#ifdef __cplusplus
extern "C" {
#endif


void nms_ongpu(const Dtype * src_ptr, Dtype *dst_ptr, const int num, const int height, const int width,
                const int num_parts, const int max_peaks, const Dtype threshold, int * work_ptr, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif
