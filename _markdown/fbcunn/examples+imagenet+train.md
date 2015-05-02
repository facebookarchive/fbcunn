

### train.lua ###

Copyright (c) 2014, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.


<a class="entityLink" href="https://github.com/facebook/fbcunn/blob/340a7c2261d022dfda11da1ac42e50b4c6819442/examples/imagenet/train.lua#L74">[src]</a>
<a name="fbcunn.train"></a>


### fbcunn.train() ###

3. train - this function handles the high-level training loop,
i.e. load data, train model, save model and state to disk

<a class="entityLink" href="https://github.com/facebook/fbcunn/blob/340a7c2261d022dfda11da1ac42e50b4c6819442/examples/imagenet/train.lua#L168">[src]</a>
<a name="fbcunn.trainBatch"></a>


### fbcunn.trainBatch(inputsThread, labelsThread) ###

4. trainBatch - Used by train() to train a single batch after the data is loaded.
