# Efficient GAN

Efficient GAN network by using PyTorch

## About CONFIG file

CONFIG files have to be written in the following format

```.yaml
train_csv_file: ./csv/train.csv

input_size: 64
channel: 1

# the hyper parameters of EGBAD
z_dim: 20
nef: 128
ngf: 128
ndf: 64

batch_size: 10

num_epochs: 500
num_fakeimg: 10

# for wandb
name: first

# training output
save_dir: ./weights
```

## Attention

- the network was checked to work properly in only the following image-size

```.yaml
channel : 1, 3
height, width  : 64
```