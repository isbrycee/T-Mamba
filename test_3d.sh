export CUDA_VISIBLE_DEVICES=0
# for Vim
python3 ./test.py --dataset 3D-CBCT-Tooth \
                        --model TMamba3D --dimension 3d \
                        --scaling_version TINY \
                        --pretrain_weight /home/jinghao/projects/T-Mamba/T-Mamba/pretrain/best_TMamba3D.pth
# for U-Mamba
# /root/paddlejob/workspace/env_run/output/haojing08/env_for_hj/minigptv2/bin/python ./test.py --dataset 3D-CBCT-Tooth \
#                         --model DenseVNet --dimension 3d \
#                         --scaling_version TINY \
#                         --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-02-01-14-41-10_DenseVNet_3D-CBCT-Tooth/checkpoints/best_DenseVNet.pth


