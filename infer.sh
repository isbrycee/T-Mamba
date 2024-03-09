/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python ./inference.py \
        --dataset 3D-CBCT-Tooth --model DenseVNet \
        --pretrain_weight ./pretrain/DenseVNet_Tooth.pth --dimension 3d \
        --scaling_version TINY --image_path ./images/1001250407_20190923.nii.gz

