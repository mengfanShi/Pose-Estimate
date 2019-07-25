void resize_forward(
	THCudaTensor *input,
	THCudaTensor *output,
	const int targetSpatialHeight, const int targetSpatialWidth, const float start_scale, const float scale_gap );