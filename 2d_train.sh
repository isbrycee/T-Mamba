export CUDA_VISIBLE_DEVICES=3
# for Vim-3D
# CUDA_VISIBLE_DEVICES=5,6 /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python -m torch.distributed.launch --nproc_per_node=2 \
# /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
#                         ./train.py --dataset 3D-CBCT-Tooth \
#                         --model DenseVNet --dimension 3d \
#                         --scaling_version TINY --epoch 20
# for Vim-2D
/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
                        ./train.py --dataset Tooth2D-X-Ray-6k \
                        --dimension 2d --model DenseVNet2D \
                        --scaling_version DEEPER_3 --epoch 30 \
                        --multi_gpu false
