# Set the base directory relative to this script's location
BASE_DIR=$(dirname "$(dirname "$0")")

export CUDA_VISIBLE_DEVICES=0
python "$BASE_DIR/inference.py" \
        --dataset 3D-CBCT-Tooth  --model SwinUNETR \
        --pretrain_weight "$BASE_DIR/pretrain/SwinUNETR_Tooth.pth --dimension 3d" \
        --scaling_version TINY \
        --image_dir "$BASE_DIR/datasets/NC-release-data-checked/valid/images" \
        --save_dir "$BASE_DIR/Result/test/SwinUNETR"
        
        #--image_dir /root/work_dir/PMFSNet/datasets/NC-release-data-checked/valid/images
        #--image_path ./images/1001250407_20190923.nii.gz \