//#ifdef __cplusplus
//extern "C" {
//#endif

#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <math.h>

#include "nms_layer_kernel.h"
#include "math_functions.h"

#define NUMBER_THREADS_PER_BLOCK_1D 16
#define NUMBER_THREADS_PER_BLOCK 256

int updiv2(const int a, const int b){
    return (a+b-1)/b;
}

__global__ void nms_register_kernel(const Dtype* const src_pointer, int* workspace, const int w, const int h, const Dtype threshold) {
	// get pixel location (x,y)
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if( x>0 && x<(w-1) && y>0 && y<(h-1) ){
		const Dtype value = src_pointer[y*w + x];
		if(value > threshold){
			const Dtype top    = src_pointer[(y-1)*w + x];
			const Dtype bottom = src_pointer[(y+1)*w + x];
			const Dtype left   = src_pointer[y*w + (x-1)];
			const Dtype right  = src_pointer[y*w + (x+1)];
			const Dtype top_left = src_pointer[(y-1)*w + x-1];
			const Dtype top_right = src_pointer[(y-1)*w + x+1];
			const Dtype bottom_left = src_pointer[(y+1)*w + x-1];
			const Dtype bottom_right = src_pointer[(y+1)*w + x+1];

			if(value > top && value > bottom && value > left && value > right && value > top_left
				&& value > bottom_left && value > bottom_right && value > top_right ){
				workspace[y*w + x] = 1;
			}
			else {
				workspace[y*w + x] = 0;
			}
		}
		else {
			workspace[y*w + x] = 0;
		}
	}	else if( x==0 || x==(w-1) || y==0 || y==(h-1) ){
		workspace[y*w + x] = 0;
	}
}


__global__ void writeResultKernel(const int length, const int* const input, const Dtype* const src_pointer, Dtype* output, const int height, const int width, const int max_peaks){
    __shared__ int local[NUMBER_THREADS_PER_BLOCK+1]; // one more
    const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIdx < length){
      local[threadIdx.x] = input[globalIdx];
      if(threadIdx.x == NUMBER_THREADS_PER_BLOCK - 1 && globalIdx != length - 1){
          //last thread in the block but not globally last, load one more
          local[threadIdx.x+1] = input[globalIdx+1];
      }
      __syncthreads();
      // see difference, except the globally last one
      if(globalIdx != length - 1){
	      if(local[threadIdx.x] != local[threadIdx.x + 1]) {
	          //means A[globalIdx] == A[globalIdx + 1] as the input[globalIdx]-th repeat
	          const int peak_index = input[globalIdx]; //0-index
	          const int peak_loc = globalIdx;
	          const int peak_loc_x = peak_loc % width;
	          const int peak_loc_y = peak_loc / width;

	          if(peak_index < max_peaks){ //limitation
	            //output[input[globalIdx]] = globalIdx;

							// if (1) {
//								float x_acc = peak_loc_x;
//								float y_acc = peak_loc_y;
//								float score_acc = src_pointer[peak_loc_y*width + peak_loc_x];
								float x_acc = 0.f;
								float y_acc = 0.f;
								float score_acc = 0.f;
								// int count = 0;
								for (int dy=-3;dy<4;dy++) {
									if ((peak_loc_y+dy)>0 && (peak_loc_y+dy)<height) {
										for (int dx=-3;dx<4;dx++) {
											if ((peak_loc_x+dx)>0 && (peak_loc_x+dx)<width) {
												const float score = src_pointer[(peak_loc_y+dy)*width + peak_loc_x+dx];
												const float x = peak_loc_x+dx;
												const float y = peak_loc_y+dy;
												if (score>0) {
													x_acc += x*score;
													y_acc += y*score;
													score_acc += score;
													// count += 1;
												}
											}
										}
									}
								}

								const int output_index = (peak_index + 1) * 3;
								output[output_index] = x_acc/score_acc;
	              output[output_index + 1] = y_acc/score_acc;
	              output[output_index + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
//                  printf("%d, %d: %d, %d: %f, %f, %f, %f\n", width, height, peak_loc_x, peak_loc_y, output[output_index], output[output_index + 1], output[output_index + 2], score_acc);
//
//	              if(output[output_index + 1] == NAN || output[output_index] == NAN) {
//                    printf("NAN\n");
//	              }
							// } else {
								// const int output_index = (peak_index + 1) * 3;
	              // output[output_index] = peak_loc_x;
	              // output[output_index + 1] = peak_loc_y;
	              // output[output_index + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
							// }
	        	}
	      	}
	      }
	      else {
	        //number of peaks
	      	output[0] = input[globalIdx] < max_peaks ? input[globalIdx] : max_peaks;
	      }
    }
}


void nms_ongpu(const Dtype * src_ptr, Dtype *dst_ptr, const int num, const int height, const int width,
                const int num_parts, const int max_peaks, const Dtype threshold, int * work_ptr, cudaStream_t stream) {

    const int offset = height * width;
	const int offset_dst = (max_peaks+1)*3;

    const dim3 threadsPerBlock(NUMBER_THREADS_PER_BLOCK_1D, NUMBER_THREADS_PER_BLOCK_1D);
	const dim3 numBlocks(updiv2(width, threadsPerBlock.x), updiv2(height, threadsPerBlock.y));
    cudaError_t	err;

	for(int n = 0; n < num; n++){ // batch
		for(int c = 0; c < num_parts; c++){
			int* w_pointer1 = work_ptr + n * num_parts * offset + c * offset;
			const Dtype* src = src_ptr + n * num_parts * offset + c * offset;
			Dtype* dst = dst_ptr + n * num_parts * offset_dst + c * offset_dst;

			// This returns w_pointer1, a binary array with 0s & 1s. 1s in the local maximum positions (size = size(src))
			nms_register_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(src, w_pointer1, width, height, threshold);
			//[0,0,0,0,1,0,0,0,0,1,0,0,0,0]
		    err = cudaGetLastError();
            if ( cudaSuccess != err )
            {
                fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
                exit( -1 );
            }

			thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(w_pointer1);

			// This modifies w_pointer1, now it indicates the local maximum indexes. Format: 0,0,0,1,1,1,1,2,2,2,... First maximum: 2, second: 6, etc...
			thrust::exclusive_scan(dev_ptr, dev_ptr + offset, dev_ptr);
			//[0,0,0,0,0,1,1,1,1,1,2,2,2,2]

			// This returns dst, with the NMS applied over it
			writeResultKernel<<<updiv2(offset,NUMBER_THREADS_PER_BLOCK), NUMBER_THREADS_PER_BLOCK, 0, stream>>>(
			                        offset, w_pointer1, src, dst, height, width, max_peaks);

		    err = cudaGetLastError();
            if ( cudaSuccess != err )
            {
                fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
                exit( -1 );
            }
		}
	}

}

//#ifdef __cplusplus
//}
//#endif
