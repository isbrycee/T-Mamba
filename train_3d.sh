export CUDA_VISIBLE_DEVICES=0
# for Vim
# CUDA_VISIBLE_DEVICES=5,6 /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python -m torch.distributed.launch --nproc_per_node=2 \
python ./train.py --dataset 3D-CBCT-Tooth \
                        --model TMamba3D --dimension 3d \
                        --pretrain_weight /home/jinghao/projects/T-Mamba/T-Mamba/pretrain/best_TMamba3D.pth \
                        --scaling_version TINY --epoch 20
# for U-Mamba
# /root/paddlejob/workspace/env_run/output/haojing08/env_for_hj/minigptv2/bin/python ./train.py --dataset 3D-CBCT-Tooth \
#                         --model DenseVNet --dimension 3d \
#                         --scaling_version TINY --epoch 20
