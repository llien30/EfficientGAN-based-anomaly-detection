# Efficient GAN

Efficient GAN network by using PyTorch


### About CONFIG file
CONFIG files have to be written in the following format
```
input_size: 64
channel: 1

#the hyper parameters of EGBAD
z_dim: 100
nef: 128
ngf: 128
ndf: 64

batch_size: 100
```

### :exclamation: Attention
- expected image size is ...
```
channel : 1
height, width  : 64(the multiple of 16 will be OK.)
```