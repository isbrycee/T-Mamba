export CUDA_VISIBLE_DEVICES=7

/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python ./test.py --dataset Tooth2D-X-Ray-6k \
                        --model TMamba2D \
                        --dimension 2d \
                        --scaling_version SMALL \
                        --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-04-18-15-52-25_TMamba2D_Tooth2D-X-Ray-6k/checkpoints/best_TMamba2D.pth