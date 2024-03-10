## Install
```sh
conda create -n tmamba python=3.9
conda activate tmamba
pip install -r requirements.txt

mamba_ssm==1.0.1
causal_conv1d==1.0.0

cd Vim-main
pip install -e mamba
cd causal-conv1d
python setup.py install
```

## Training
```sh
sh train.sh
```

## Testing
```sh
sh test.sh
```
