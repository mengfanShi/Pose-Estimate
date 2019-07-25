#include <TH/TH.h>
#include <stdio.h>
#include "limbs_coco.h"

void limbs_coco_cpu(THFloatTensor * heatmap, THFloatTensor * peaks, THFloatTensor * joints, THIntTensor *num_joints,
                const int connect_min_subset_cnt, const float connect_min_subset_score,
                const float connect_inter_threshold, const int connect_inter_min_above_threshold)
{
    int is_batch = 1;
    if (heatmap->nDimension == 3) {
        /* Force batch */
        is_batch = 0;
        THFloatTensor_resize4d(heatmap, 1, heatmap->size[0], heatmap->size[1], heatmap->size[2]);
        THFloatTensor_resize4d(peaks, 1, peaks->size[0], peaks->size[1], peaks->size[2]);
    }

    int batch_size = heatmap->size[0];
//    if(batch_size > 1) {
//        printf("limbs_cpu only support batch_size=1 now");
//        exit(-1);
//    }

    int n_parts = peaks->size[1];
    int max_peaks = peaks->size[2] - 1;

    int height = heatmap->size[2];
    int width = heatmap->size[3];

//    float joints[MAX_NUM_PARTS*3*MAX_PEOPLE];
    THFloatTensor_resize4d(joints, batch_size, MAX_PEOPLE, n_parts, 3);
    THFloatTensor_zero(joints);

    THIntTensor_resize1d(num_joints, batch_size);
    int *ptr_cnts = THIntTensor_data(num_joints);

//    const int connect_min_subset_cnt = 3;
//    const float connect_min_subset_score = 0.4;
//    const float connect_inter_threshold = 0.050;
//    const int connect_inter_min_above_threshold = 9;
    int elt;
#pragma omp parallel for
    for (elt = 0; elt < batch_size; elt++) {
        /* Helpers */
        THFloatTensor *heatmap_n = THFloatTensor_newSelect(heatmap, 0, elt);
        THFloatTensor *peaks_n = THFloatTensor_newSelect(peaks, 0, elt);
        THFloatTensor *joints_n = THFloatTensor_newSelect(joints, 0, elt);

        int cnt = connectLimbsCOCO(
            THFloatTensor_data(heatmap_n),
            THFloatTensor_data(peaks_n),
            max_peaks,
            THFloatTensor_data(joints_n),
            height, width,
            connect_min_subset_cnt, connect_min_subset_score,
            connect_inter_threshold, connect_inter_min_above_threshold);

        ptr_cnts[elt] = cnt;

        /* Free */
        THFloatTensor_free( heatmap_n );
        THFloatTensor_free(peaks_n);
        THFloatTensor_free(joints_n);
    }

    if (is_batch == 0) {
        THFloatTensor_resize3d(heatmap, heatmap->size[1], heatmap->size[2], heatmap->size[3]);
        THFloatTensor_resize3d(peaks, peaks->size[1], peaks->size[2], peaks->size[3]);
    }
}

