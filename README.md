# <img src="images/tmamba_logo.jpeg" width="150"> 

# T-Mamba

[Jing Hao](https://scholar.google.com/citations?user=E8R8c00AAAAJ&hl=zh-CN), [Lei He](), [Kuo Feng Hung](https://scholar.google.com/citations?user=17V5x14AAAAJ&hl=zh-CN).

This repository is the official implementation of the [T-Mamba: Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation](https://arxiv.org/pdf/2404.01065.pdf).

More experiments are running 🔥

Currently, our T-Mamba supports 2D & 3D vision tasks. Welcome to try it for improving your model's performance.

Coming soon...

If u have any quesitons, pls feel free to drop me via isjinghao@gmail.com.

## Install
```sh
conda create -n tmamba python=3.9
conda activate tmamba
pip install -r requirements.txt

cd ../causal-conv1d
python setup.py install
cd Vim-main/mamba
python setup.py install

=============================
Requirement specific version:
mamba_ssm==1.0.1
causal_conv1d==1.0.0
=============================
```

## Training
```sh
sh train.sh # for 3D
sh 2d_train.sh # for 2D
```

## Testing
```sh
sh test.sh
```

## Inference
```sh
sh infer.sh
```
