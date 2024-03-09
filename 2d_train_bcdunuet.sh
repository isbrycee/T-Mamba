export CUDA_VISIBLE_DEVICES=2

# for Vim-2D
/root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
                        ./train.py --dataset Tooth2D-X-Ray-6k \
                        --model BCDUNet --dimension 2d \
                        --scaling_version TINY --epoch 30 \
                        --multi_gpu false

