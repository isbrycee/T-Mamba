## Install
```sh
conda create -n tmamba python=3.9
conda activate tmamba
pip install -r requirements.txt

cd Vim-main
pip install -e mamba
pip install -e causal_conv1d>=1.1.0
```

## Training
```sh
sh train.sh
```

## Testing
```sh
sh test.sh
'''
