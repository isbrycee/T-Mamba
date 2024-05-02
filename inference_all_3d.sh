#!/bin/bash

# Run each script sequentially
./inference_bash_script/Ours.sh
./inference_bash_script/DenseVNet.sh
./inference_bash_script/nnFormer.sh
./inference_bash_script/PMFSNet.sh
./inference_bash_script/SwinUNETR.sh


# Run each script sequentially
if ! ./Ours.sh; then
    echo "Ours.sh failed"
    exit 1
fi

if ! ./DenseVNet.sh; then
    echo "DenseVNet.sh failed"
    exit 1
fi

if ! ./nnFormer.sh; then
    echo "nnFormer.sh failed"
    exit 1
fi

if ! ./PMFSNet.sh; then
    echo "PMFSNet.sh failed"
    exit 1
fi

if ! ./SwinUNETR.sh; then
    echo "SwinUNETR.sh failed"
    exit 1
fi
