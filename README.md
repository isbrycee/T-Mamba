<p align="center">
  <img src="images/T-Mamba-logo.png" width="150">
</p>


# T-Mamba

[Jing Hao](https://scholar.google.com/citations?user=E8R8c00AAAAJ&hl=zh-CN), [Lei He](), [Kuo Feng Hung](https://scholar.google.com/citations?user=17V5x14AAAAJ&hl=zh-CN).

This repository is the official implementation of the [T-Mamba: A unified framework with Long-Range Dependency in dual-domain for 2D & 3D Tooth Segmentation
](https://arxiv.org/pdf/2404.01065.pdf).

We have conducted more experiments and analysis on 3D CBCT and 2D X-ray images, and have updated the whole manuscript. 🏃‍♂️

The code, pre-trained weights, and datasets is fully available.

Currently, our T-Mamba supports 2D & 3D vision tasks. Welcome to try it for improving your model's performance. 🔥

The proposed TED3 dataset is available at: [Hugging Face](https://huggingface.co/datasets/Bryceee/TED3).

If u have any quesitons, pls feel free to drop me via isjinghao@gmail.com.

## Install
```sh
conda create -n tmamba python=3.9
conda activate tmamba
pip install -r requirements.txt

cd Tim/causal-conv1d
python setup.py install
cd ../mamba
python setup.py install

=============================
Requirement specific version:
mamba_ssm==1.0.1
causal_conv1d==1.0.0
=============================
```

## Training
```sh
sh train_3d.sh # for 3D
sh train_2d.sh # for 2D
```

## Testing (for evaluations)
```sh
sh test_3d.sh # for 3D
sh test_2d.sh # for 2D
```

## Inference
```sh
sh infer_3d.sh # for 3D
sh infer_2d.sh # for 2D
```

## Citing SAM 2

If you use TED3 dataset or the T-Mamba network in your research, please use the following BibTeX entry.

```bibtex
@article{hao2024t,
  title={T-Mamba: A unified framework with Long-Range Dependency in dual-domain for 2D \& 3D Tooth Segmentation},
  author={Hao, Jing and Zhu, Yonghui and He, Lei and Liu, Moyun and Tsoi, James Kit Hon and Hung, Kuo Feng},
  journal={arXiv preprint arXiv:2404.01065},
  year={2024}
}
```
