# for multiple gpus
export CUDA_VISIBLE_DEVICES=1,2,3
CUDA_VISIBLE_DEVICES=1,2,3 /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
                     -m torch.distributed.launch \
                     --master_port=25679 \
                     --nproc_per_node=3 \
                     ./train.py --dataset Tooth2D-X-Ray-6k \
                        --dimension 2d --model TMamba2D \
                        --scaling_version SMALL --epoch 30 \
                        --multi_gpu True

# for single gpu
# /root/paddlejob/workspace/env_run/output/haojing08/minigptv2/bin/python \
#                         ./train.py --dataset Tooth2D-X-Ray-6k \
#                         --dimension 2d --model TMamba2D \
#                         --scaling_version SMALL --epoch 30 \
#                         --multi_gpu False