##Training an Object Classifier in Torch-7 on multiple GPUs over [ImageNet](http://image-net.org/download-images)

In this concise example (1200 lines including a general-purpose and highly scalable data loader for images), we showcase:
- train [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) on ImageNet
- use nn.ModelParallel to speedup training over multiple GPUs
- use nn.SpatialConvolutionCuFFT to speedup training even more
- multithreaded data-loading from disk (showcases sending tensors from one thread to another without serialization)

### Requirements
- Install Torch-7, graphicsmagick, nn, cunn, cudnn, fbnn, fbcunn, threads (luarocks install [package])
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

### Running
To run the training, simply run main.lua
```bash
th main.lua --data [imagenet-folder with train and val folders]
```

To get help and command-line options, use --help
```bash
th main.lua --help
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