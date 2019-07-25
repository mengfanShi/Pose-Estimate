#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "math_functions.h"
#include "imresize_layer_kernel.h"


#define NUMBER_THREADS_PER_BLOCK_1D 16
#define NUMBER_THREADS_PER_BLOCK 256

int updiv1(const int a, const int b){
    return (a+b-1)/b;
}

inline __device__ void cubic_interpolation(Dtype &out, const Dtype &v0, const Dtype &v1, const Dtype &v2, const Dtype &v3, const float dx) {
    // Dtype a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // Dtype b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // Dtype c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}


__global__ void imresize_cubic_kernel(const Dtype* const src_ptr, Dtype* dst_pointer, const int src_offset, const int num, const float scale_gap,
									                    const float start_scale, const int oriSpatialWidth, const int oriSpatialHeight, const int tw, const int th){
	// get pixel location (x,y)
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// begin compute
	if(x < tw && y < th) {
		Dtype d_temp = 0;
		Dtype sum = 0;
		for(int n = 0; n < num; n++){
			const int padw = floor(oriSpatialWidth /2 * (1-start_scale + n * scale_gap) ); //n
			const int padh = floor(oriSpatialHeight /2 * (1-start_scale + n * scale_gap) );
			const int ow = oriSpatialWidth - 2*padw;
			const int oh = oriSpatialHeight - 2*padh;
			//LOG(ERROR) << "GPU padw " << padw << " padh " << padh;
			const Dtype* const src_pointer = src_ptr + n * src_offset;

			const float offset_x = tw/float(ow)/2 - 0.5;
			const float offset_y = th/float(oh)/2 - 0.5;
			const float x_on_ori = (x - offset_x) * (float(ow) / tw);  //3.5 is for 8x enlarge
			const float y_on_ori = (y - offset_y) * (float(oh) / th);

			int x_nei[4];
			x_nei[1] = int(x_on_ori + 1e-5);
			x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
			x_nei[0] = ((x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1)) + padw;
			x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
			x_nei[3] = ((x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1)) + padw;
			const float dx = x_on_ori - x_nei[1];
			x_nei[1] = x_nei[1] + padw;
			x_nei[2] = x_nei[2] + padw;

			int y_nei[4];
			y_nei[1] = int(y_on_ori + 1e-5);
			y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
			y_nei[0] = ((y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1)) + padh;
			y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
			y_nei[3] = ((y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1)) + padh;
			const float dy = y_on_ori - y_nei[1];
			y_nei[1] = y_nei[1] + padh;
			y_nei[2] = y_nei[2] + padh;

			Dtype temp[4];
			for(int i = 0; i < 4; i++){
				cubic_interpolation(temp[i], src_pointer[y_nei[i]*(ow+2*padw) + x_nei[0]],
					                         src_pointer[y_nei[i]*(ow+2*padw) + x_nei[1]],
					                         src_pointer[y_nei[i]*(ow+2*padw)+ x_nei[2]],
					                         src_pointer[y_nei[i]*(ow+2*padw) + x_nei[3]], dx);
			}
			//cubic_interpolation(dst_pointer[y*tw+x], temp[0], temp[1], temp[2], temp[3], dy);
			cubic_interpolation(d_temp, temp[0], temp[1], temp[2], temp[3], dy);
			sum = sum + d_temp;
		}
		dst_pointer[y*tw+x] = sum / num;
	}
}


void imresize_ongpu(const Dtype * src_pointer, Dtype *dst_pointer, const int num, const int channel, const int height, const int width,
                const int targetSpatialHeight, const int targetSpatialWidth, const float start_scale, const float scale_gap, cudaStream_t stream)
{
    const dim3 threadsPerBlock(NUMBER_THREADS_PER_BLOCK_1D, NUMBER_THREADS_PER_BLOCK_1D);
	const dim3 numBlocks(updiv1(targetSpatialWidth, threadsPerBlock.x), updiv1(targetSpatialHeight, threadsPerBlock.y));
	const int offset_src = height * width;
	const int offset_dst = targetSpatialWidth * targetSpatialHeight;

    for(int c = 0; c < channel; c++){
        // imresize_cubic_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + (n * channel + c) * offset_src,
        // 	                                                     dst_pointer + (n * channel + c) * offset_dst,
        // 	                                    			           oriSpatialWidth, oriSpatialHeight,
        // 	                                    			           targetSpatialWidth, targetSpatialHeight);
        imresize_cubic_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(  src_pointer + c * offset_src,
                                                                dst_pointer + c * offset_dst,
                                                                channel* offset_src, num, scale_gap, start_scale,
                                                                width, height,
                                                                targetSpatialWidth, targetSpatialHeight);

    }


}



#ifdef __cplusplus
}
#endif