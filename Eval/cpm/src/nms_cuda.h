void cpm_nms_forward(
	THCudaTensor *input,
	THCudaTensor *output,
	const int num_parts, const int max_peaks, const float threshold);