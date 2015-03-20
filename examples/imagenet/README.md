##Training an Object Classifier in Torch-7 on multiple GPUs over [ImageNet](http://image-net.org/download-images)

In this concise example (1200 lines including a general-purpose and highly scalable data loader for images), we showcase:
- train [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) or [Overfeat](http://arxiv.org/abs/1312.6229) on ImageNet
- showcase multiple backends: CuDNN, CuNN, FBCuNN
- use nn.ModelParallel and nn.DataParallel to speedup training over multiple GPUs
- use nn.SpatialConvolutionCuFFT to speedup training even more
- multithreaded data-loading from disk (showcases sending tensors from one thread to another without serialization)

### Requirements
- Install everything needed using the commands here: [INSTALL.md](../../INSTALL.md)
- Download Imagenet-12 dataset from http://image-net.org/download-images . It has 1000 classes and 1.2 million images.

### Data processing
**The images dont need to be preprocessed or packaged in any database.** It is preferred to keep the dataset on an [SSD](http://en.wikipedia.org/wiki/Solid-state_drive) but we have used the data loader comfortably over NFS without loss in speed.
We just use a simple convention: SubFolderName == ClassName.
So, for example: if you have classes {cat,dog}, cat images go into the folder dataset/cat and dog images go into dataset/dog

The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968).
You need to get the validation groundtruth and move the validation images into appropriate subfolders.
To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands:
```bash
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

Now you are all set!

If your imagenet dataset is on HDD or a slow SSD, run this command to resize all the images such that the smaller dimension is 256 and the aspect ratio is intact.
This helps with loading the data from disk faster.
```bash
find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}
```

### Running
The training scripts come with several options which can be listed by running the script with the flag --help
```bash
th main.lua --help
```

To run the training, simply run main.lua
By default, the script runs 1-GPU AlexNet with the CuDNN backend and 2 data-loader threads.
```bash
th main.lua -data [imagenet-folder with train and val folders]
```

For 2-GPU model parallel AlexNet + CuDNN, you can run it this way:
```bash
th main.lua -data [imagenet-folder with train and val folders] -nGPU 2 -backend cudnn -netType alexnet
```
Similarly, you can switch the backends to 'fbcunn' or 'cunn' to use a different set of CUDA kernels.
Using 'fbcunn' will run faster, at the expense of a little extra memory

You can also alternatively train OverFeat using this following command:
```bash
th main.lua -data [imagenet-folder with train and val folders] -netType overfeat

# multi-GPU overfeat (let's say 2-GPU)
th main.lua -data [imagenet-folder with train and val folders] -netType overfeat -nGPU 2
```

The training script prints the current Top-1 and Top-5 error as well as the objective loss at every mini-batch.
We hard-coded a learning rate schedule so that AlexNet converges to an error of 42.5% at the end of 53 epochs.

At the end of every epoch, the model is saved to disk (as model_[xx].t7 where xx is the epoch number).
You can reload this model into torch at any time using torch.load
```lua
model = torch.load('model_10.t7') -- loading back a saved model
```

Similarly, if you would like to test your model on a new image, you can use testHook from line 103 in donkey.lua to load your image, and send it through the model for predictions. For example:
```lua
dofile('donkey.lua')
img = testHook({loadSize}, 'test.jpg')
model = torch.load('model_10.t7')
model:evaluate()
predictions = model:forward(img:cuda())
```

If you ever want to reuse this example, and debug your scripts, it is suggested to debug and develop in the single-threaded mode, so that stack traces are printed fully.
```lua
th main.lua -nDonkeys 0 [...options...]
```

### Code Description
- `main.lua` (~30 lines) - loads all other files, starts training.
- `opts.lua` (~50 lines) - all the command-line options and description
- `data.lua` (~60 lines) - contains the logic to create K threads for parallel data-loading.
- `donkey.lua` (~200 lines) - contains the data-loading logic and details. It is run by each data-loader thread. random image cropping, generating 10-crops etc. are in here.
- `model.lua` (~80 lines) - creates AlexNet model and criterion
- `train.lua` (~190 lines) - logic for training the network. we hard-code a learning rate + weight decay schedule that produces good results.
- `test.lua` (~120 lines) - logic for testing the network on validation set (including calculating top-1 and top-5 errors)
- `dataset.lua` (~430 lines) - a general purpose data loader, mostly derived from [here: imagenetloader.torch](https://github.com/soumith/imagenetloader.torch). That repo has docs and more examples of using this loader.
