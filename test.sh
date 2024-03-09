export CUDA_VISIBLE_DEVICES=3
# for Vim
/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python ./test.py --dataset 3D-CBCT-Tooth \
                        --model DenseVNet --dimension 3d \
                        --scaling_version TINY \
                        --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-02-26-11-14-08_DenseVNet_3D-CBCT-Tooth/checkpoints/best_DenseVNet.pth
# for U-Mamba
# /root/paddlejob/workspace/env_run/output/haojing08/env_for_hj/minigptv2/bin/python ./test.py --dataset 3D-CBCT-Tooth \
#                         --model DenseVNet --dimension 3d \
#                         --scaling_version TINY \
#                         --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-02-01-14-41-10_DenseVNet_3D-CBCT-Tooth/checkpoints/best_DenseVNet.pth


