python ./train.py --dataset Tooth2D-X-Ray-6k \
                         --dimension 2d --model TMamba2D \
                         --scaling_version SMALL --epoch 30 \
                         --pretrain_weight /home/jinghao/projects/T-Mamba/T-Mamba/pretrain/best_TMamba2D.pth \
                         --multi_gpu False 