void limbs_coco_cpu(THFloatTensor * heatmap, THFloatTensor * peaks, THFloatTensor * joints, THIntTensor *num_joints,
                const int connect_min_subset_cnt, const float connect_min_subset_score,
                const float connect_inter_threshold, const int connect_inter_min_above_threshold);