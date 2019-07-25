#include <THC/THC.h>
#include <stdio.h>
#include "nms_layer_kernel.h"

#define THCUNN_assertSameGPU( ... ) THAssertMsg( THCudaTensor_checkGPU( __VA_ARGS__ ), \
						 "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one." )

extern THCState *state;

void cpm_nms_forward(
	THCudaTensor *input,
	THCudaTensor *output,
	const int num_parts, const int max_peaks, const float threshold)
{
	THCUNN_assertSameGPU( state, 2, input, output );

	/* Params: */
	int num = input->size[0];
	int channel = input->size[1];
	int height = input->size[2];
	int width = input->size[3];

	/* Resize output */
	THCudaTensor_resize4d( state, output, num, num_parts, max_peaks + 1, 3 );
	THCudaTensor_zero( state, output );

	THCudaIntTensor *workspace	= THCudaIntTensor_new( state );
	THCudaIntTensor_resize4d( state, workspace, num, channel, height, width );
	THCudaIntTensor_zero( state, workspace );

	// run on gpu
    nms_ongpu(
        THCudaTensor_data( state, input ),
	    THCudaTensor_data( state, output ),
        num, height, width, num_parts, max_peaks, threshold,
        THCudaIntTensor_data( state, workspace ),
        THCState_getCurrentStream( state ));

//	input = THCudaTensor_newContiguous( state, input );
//	THCudaTensor_free( state, input );
	THCudaIntTensor_free( state, workspace );
}
