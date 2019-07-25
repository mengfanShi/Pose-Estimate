#include <THC/THC.h>
#include <stdio.h>
#include "imresize_layer_kernel.h"

#define THCUNN_assertSameGPU( ... ) THAssertMsg( THCudaTensor_checkGPU( __VA_ARGS__ ), \
						 "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one." )

extern THCState *state;

void resize_forward(
	THCudaTensor *input,
	THCudaTensor *output,
	const int targetSpatialHeight, const int targetSpatialWidth, const float start_scale, const float scale_gap )
{
	THCUNN_assertSameGPU( state, 2, input, output );

	/* Params: */
	int num = input->size[0];
	int channel = input->size[1];
	int height = input->size[2];
	int width = input->size[3];

	// resize output
	THCudaTensor_resize4d( state, output, num, channel, targetSpatialHeight, targetSpatialWidth );

	// resize on gpu
	imresize_ongpu(
	    THCudaTensor_data( state, input ),
	    THCudaTensor_data( state, output ),
	    num, channel, height, width, targetSpatialHeight, targetSpatialWidth, start_scale, scale_gap,
	    THCState_getCurrentStream( state )
	);

}
