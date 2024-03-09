export CUDA_VISIBLE_DEVICES=0
# for Vim-3D
# CUDA_VISIBLE_DEVICES=5,6 /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python -m torch.distributed.launch --nproc_per_node=2 \
# /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
#                         ./train.py --dataset 3D-CBCT-Tooth \
#                         --model DenseVNet --dimension 3d \
#                         --scaling_version TINY --epoch 20
# for Vim-2D
/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
                        ./train.py --dataset Tooth2D-X-Ray-6k \
                        --model MsRED --dimension 2d \
                        --scaling_version TINY --epoch 30 \
                        --multi_gpu false

