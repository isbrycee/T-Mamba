export CUDA_VISIBLE_DEVICES=0

python ./inference.py \
        --dataset Tooth2D-X-Ray-6k --model TMamba2D \
        --pretrain_weight /home/jinghao/projects/T-Mamba/T-Mamba/runs/2025-06-03-20-09-21_TMamba2D_Tooth2D-X-Ray-6k/checkpoints/best_TMamba2D.pth \
        --dimension 2d \
        --scaling_version SMALL \
        --is_visual \
        --save_dir /home/jinghao/projects/T-Mamba/datasets/MICCAI2023/2D/test_infer_masks/ \
        --image_dir /home/jinghao/projects/T-Mamba/datasets/MICCAI2023/2D/test_image
        # --image_path datasets/Tooth2D-X-Ray-6k-val-infer-compare/test/images/78-No004.jpg \