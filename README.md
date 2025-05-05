# Getting Started

## Environment
```
conda env create -f env.yml

pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset
D1: https://github.com/CloudWise-OpenSource/GAIA-DataSet

D1 contains two datasets: MicroSS and Companion Data. We use MicroSS, for it provides trace, log, and metric at the same time.

## Demo
We provide a demo. Please run:
```
python main.py --dataset gaia
```
