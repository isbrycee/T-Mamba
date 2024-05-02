#!/bin/bash

# Set the base directory relative to this script's location
BASE_DIR=$(dirname "$(dirname "$0")")

# Set the CUDA device for GPU processing
export CUDA_VISIBLE_DEVICES=0


# Execute the Python script for T-Mamba
python "$BASE_DIR/inference.py" \
        --dataset 3D-CBCT-Tooth \
        --model TMamba3D \
        --pretrain_weight "$BASE_DIR/pretrain/best_TMamba3D.pth" \
        --dimension 3d \
        --scaling_version TINY \
        --image_dir "$BASE_DIR/datasets/NC-release-data-checked/valid/images" \
        --save_dir "$BASE_DIR/Result/test/Ours"