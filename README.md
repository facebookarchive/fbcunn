# `fbcunn` - Deep Learning CUDA Extensions from Facebook AI Research.

## What?
[Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) is a popular kid in machine learning these days.
At [Facebook AI Research](http://research.facebook.com/ai/) we've been doing quite a bit of deep learning research.
This repository contains our highly engineered deep learning modules for GPUs, to accelerate your own deep learning endeavors.
It plugs into the [Torch-7](https://github.com/torch/torch7/wiki/Cheatsheet) framework and  installs seamlessly via `luarocks`, 
and is fully compatible with torch's [nn](https://github.com/torch/nn) package.

In summary, we're releasing fast nn modules for Convnets and neural networks in general:
- Fast spatial convolution modules that use FFT to accelerate convolutions. [We wrote a paper about them](http://arxiv.org/abs/1412.7580) if you'd like to read more.
- Fast Temporal convolutions that are 1.5x to 10x faster compared to Torch's cunn implementations.
- nn.DataParallel and nn.ModelParallel containers. Plug your model in them and see it accelerate over multiple GPUs
- Wrappers to use FFT/IFFT as nn modules.
- Fast LookupTable that is used for Neural Language Models and word embeddings. Much faster than the one in torch/nn
- Hierarchical SoftMax module, now classifying 1 million classes is a practically viable strategy
- LP and Max Pooling over feature maps (usable for MaxOut).
- more goodies. Full documentation and spec is here: https://facebook.github.io/fbcunn/fbcunn/

Examples:
- Training an imagenet based classifier in Torch-7 using multiple GPUs (showcasing our FFT convolutions as well as our ModelParallel container)

## Why?
We know that science and technology progress faster when researchers exchange ideas and tools. Making significant progress in AI will take the participation of the entire research community, and We want to do what we can to make the field progress faster. That is why we love open science and open source. We publish our research with open access, very often on [Arxiv](http://arxiv.org), on [our members' web sites](http://research.facebook.com/ai), and eventually on the [FAIR publications page](https://research.facebook.com/publications/ai/). And we share our code right here!

## Who is this for?
This will help you if you want to train large-scale deep learning systems (particularly convolutional nets) for image recognition, NLP, or other applications. This will help you particularly well if already are a Torch user.

## How to install them?
You will find step-by-step and detailed installation instructions in **[INSTALL.md](INSTALL.md)**

We've worked hard to make the install as pain-free as possible. If you have an issue, use github issues, we'll try our best to help.

## How to use them?

- The DataParallel and ModelParallel modules are super-simple to use. The unit-test doubles as both an example as well as a test. There is also a practical example of ModelParallel in examples/imagenet. If you want more examples, please do ask.
```lua
m = nn.DataParallel():add(nn.SpatialConvolution(...)):add(nn.ReLU()) -- see, so simple
```

- Convolution modules are even simpler to use. They are fully API compatible with their [nn equivalents](https://github.com/torch/nn/blob/master/doc/convolution.md). For an example, look at examples/imagenet
```lua
conv = nn.SpatialConvolutionCuFFT(...) -- fast spatial convolutions!
conv = nn.TemporalConvolutionFB(...) -- fast temporal convolutions!
```

- LookupTable is named `nn.LookupTableGPU` and Hierarchical SoftMax as `nn.HSM`, they are super-simple to use as well, check the docs out.

https://facebook.github.io/fbcunn/fbcunn/

The unit tests in the test/ folder also double as examples! If you have a question, do ask.


## I want exact details of everything...

API docs, generated with [torch-dokx](https://github.com/deepmind/torch-dokx), are available at http://facebook.github.io/fbcunn/fbcunn/

Some of the unit tests need [fbnn](https://github.com/facebook/fbnn)

## License

`fbcunn` is BSD-licensed. We also provide an additional patent
grant.
