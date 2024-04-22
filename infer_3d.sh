export CUDA_VISIBLE_DEVICES=0

/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python ./inference.py \
        --dataset 3D-CBCT-Tooth --model TMamba3D \
        --pretrain_weight /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/best_DenseVNet.pth --dimension 3d \
        --scaling_version TINY \
        --image_path ./images/1001250407_20190923.nii.gz \
        --image_dir /root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets/NC-release-data-checked/valid/images

