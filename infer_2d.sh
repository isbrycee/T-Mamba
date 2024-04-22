export CUDA_VISIBLE_DEVICES=1

/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python ./inference.py \
        --dataset Tooth2D-X-Ray-6k --model SwinUMamba \
        --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-04-14-13-18-01_SwinUMamba_Tooth2D-X-Ray-6k/checkpoints/best_SwinUMamba.pth \
        --dimension 2d \
        --scaling_version SMALL \
        --image_path datasets/Tooth2D-X-Ray-6k-val-infer-compare/test/images/78-No004.jpg \
        --is_visual \
        --save_dir /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k-val-infer-compare/SwinUMamba/ \
        --image_dir /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/Tooth2D-X-Ray-6k-val-infer-compare/test/images