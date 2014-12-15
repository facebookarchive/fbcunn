# `fbcunn` - Facebook's deeplearning CUDA extensions.

## What?
[Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) is a popular kid in machine learning these days.
At [Facebook AI Research](http://research.facebook.com/ai/) we've been doing quite a bit of deep learning research.
This repository contains our highly engineered deep learning modules for GPUs, to accelerate your own deep learning endeavours.
It plugs into the [Torch-7](https://github.com/torch/torch7/wiki/Cheatsheet) framework and installs seamlessly via `luarocks`, 
and is fully compatible with torch's [nn](https://github.com/torch/nn) package.

In summary, we're releasing fast nn modules for Convnets and neural networks in general:
- Fast spatial convolution module that uses FFT to accelerate convolutions. [We wrote a paper about them](http://arxiv.org/abs/1412.7580) if you'd like to read more.
- Fast Temporal convolutions that are 1.5x to 10x faster compared to Torch's cunn implementations.
- nn.DataParallel and nn.ModelParallel containers. Plug your model in them and see it accelerate over multiple GPUs
- Wrappers to use FFT/IFFT as nn modules.
- Fast LookupTable that is used for Neural Language Models and word embeddings. Much faster than the one in torch/nn
- Hierarchical SoftMax module, now classifying 1 million classes is a practically viable strategy
- LP and Max Pooling over feature maps.
- more goodies. Full documentation and spec is here: https://facebook.github.io/fbcunn/fbcunn/

## Why?
We love open science and open source. In short, we love sharing.
[We publish our research there](https://research.facebook.com/publications/ai/). 
We share our code right here!

## Who is this for?
You use Torch for your neural networks and deep learning research, or you work with another machine learning package but you like fast code regardless.

## How to install them?
- Find a machine with [Ubuntu 14.04+](http://www.ubuntu.com/) and an NVIDIA GPU with compute capability 3.5 or above, as well as CUDA 6.5.
- [Install Torch-7](https://github.com/torch/torch7/wiki/Cheatsheet). It's awesome.
- Install folly, fbthrift, thpp and fblualib by running [this simple script](https://github.com/soumith/fblualib/blob/master/install_all.sh). Take a lunch break, this takes some time, compiling and all...
- That's it, now install fbcunn:
```bash
git clone https://github.com/soumith/fbcunn.git
cd fbcunn && luarocks make rocks/fbcunn-scm-1.rockspec # got get a coffee
```
We've worked hard to make the install as pain-free as possible. If you have an issue, use github issues, we'll try our best to help.

## How to use them?

- The DataParallel and ModelParallel modules are super-simple to use. The unit-test doubles as both an example as well as a test. If you want more examples, please do ask.
```lua
m = nn.DataParallel():add(nn.SpatialConvolution(...)):add(nn.ReLU()) -- see, so simple
```

- Convolution modules are even simpler to use. They are fully API compatible with their [nn equivalents](https://github.com/torch/nn/blob/master/doc/convolution.md)
```lua
conv = nn.SpatialConvolutionCuFFT(...) -- fast spatial convolutions!
conv = nn.TemporalConvolutionFB(...) -- fast temporal convolutions!
```

- LookupTable is named `nn.LookupTableGPU` and Hierarchical SoftMax as `nn.HSM`, they are super-simple to use as well, check the docs out.

https://facebook.github.io/fbcunn/fbcunn/

The unit tests in the test/ folder also double as examples! If you have a question, do ask.


## I want exact details of everything...
API docs - https://facebook.github.io/fbcunn/fbcunn/

Some of the unit tests need [fbnn](https://github.com/facebook/fbnn)

## License

`fbcunn` is BSD-licensed. We also provide an additional patent
grant.
