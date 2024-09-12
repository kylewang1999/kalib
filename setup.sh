#!/bin/bash

#############################################
# Download the pretrained checkpoints#

# MiDaS Checkpoint: https://github.com/isl-org/MiDaS?tab=readme-ov-file#accuracy
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt -P ./pretrained_checkpoints/

# ZoeDepth Checkpoint: https://github.com/isl-org/ZoeDepth/releases
wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt -P ./pretrained_checkpoints/

# CoTracker Checkpoint: https://huggingface.co/facebook/cotracker
# wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth -P ./pretrained_checkpoints/
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth -P ./pretrained_checkpoints/
#############################################

# easyhec_repo_path=$PWD/third_party/easyhec/
# grounded_sam_repo_path=$PWD/third_party/grounded_segment_anything/
# spatial_tracker_repo_path=$PWD/third_party/spatial_tracker/
# dataset_dir=<your dataset dir>
# python easycalib_demo.py\
#     --root_dir $dataset_dir \
#     --use_segm_mask true \
#     --caliberate_method pnp \
#     --pnp_refinement true \
#     --use_pnp_ransac false \
#     --use_grounded_sam 
#     --has_gt \
#     --win_len 1 \
#     --verbose \
#     --render_mask \
#     --easyhec_repo_path $easyhec_repo_pah \
#     --grounded_sam_repo_path $grounded_sam_repo_path \
#     --spatial_tracker_repo_path $spatial_tracker_repo_path \
#     --cut_off 300 \P
#     --renderer_device_id 0 \
#     --tracking_device_id 0 \
#     --mask_inference_device_id 0 \
#     --keypoint_ids 0