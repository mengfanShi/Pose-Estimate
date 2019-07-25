#ifndef _CPP_Test
#define _CPP_Test

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PEOPLE 96
#define MAX_NUM_PARTS 70

int connectLimbsCOCO(
    const float *heatmap_pointer,
    const float *in_peaks,
    int max_peaks,
    float *joints,
    const int NET_RESOLUTION_WIDTH, const int NET_RESOLUTION_HEIGHT,
    const int connect_min_subset_cnt, const float connect_min_subset_score,
    const float connect_inter_threshold, const int connect_inter_min_above_threshold);


#ifdef __cplusplus
}
#endif

#endif