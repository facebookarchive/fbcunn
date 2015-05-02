

### AbstractParallel.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbcunn.AbstractParallel.dok"></a>


## fbcunn.AbstractParallel ##


`nn.AbstractParallel` is the base class for modules controlling
data/model-parallel behaviour in Torch.

The key concept is that data/model-parallelism _splits_ along a
dimension, and this class controls the distribution of input and
merging of output along this dimension.

To extend this class, override `_distributeInput` as appropriate.

See `nn.DataParallel` and `nn.ModelParallel` for examples of usage.


<a class="entityLink" href="https://github.com/facebook/fbcunn/blob/340a7c2261d022dfda11da1ac42e50b4c6819442/fbcunn/AbstractParallel.lua#L52">[src]</a>
<a name="fbcunn.AbstractParallel:nextGPU"></a>


### fbcunn.AbstractParallel:nextGPU() ###


This function yields the GPU id for the module to be added.

It can be used for load balancing. It assumes all GPUs are available.


<a class="entityLink" href="https://github.com/facebook/fbcunn/blob/340a7c2261d022dfda11da1ac42e50b4c6819442/fbcunn/AbstractParallel.lua#L83">[src]</a>
<a name="fbcunn.AbstractParallel:gpuSend"></a>


### fbcunn.AbstractParallel:gpuSend(dest, source) ###


Asynchronous copy from dest to source.

Use with caution; there needs to be some sort of external synchronization to
prevent source from being modified after this copy is enqueued.



#### Undocumented methods ####

<a name="fbcunn.AbstractParallel"></a>
 * `fbcunn.AbstractParallel(dimension)`
<a name="fbcunn.AbstractParallel:add"></a>
 * `fbcunn.AbstractParallel:add(module, gpuid)`
<a name="fbcunn.AbstractParallel:get"></a>
 * `fbcunn.AbstractParallel:get(index)`
<a name="fbcunn.AbstractParallel:updateOutput"></a>
 * `fbcunn.AbstractParallel:updateOutput(input)`
<a name="fbcunn.AbstractParallel:updateGradInput"></a>
 * `fbcunn.AbstractParallel:updateGradInput(_input, gradOutput)`
<a name="fbcunn.AbstractParallel:accGradParameters"></a>
 * `fbcunn.AbstractParallel:accGradParameters(_input, _gradOutput, scale)`
<a name="fbcunn.AbstractParallel:accUpdateGradParameters"></a>
 * `fbcunn.AbstractParallel:accUpdateGradParameters(_input, _gradOutput, lr)`
<a name="fbcunn.AbstractParallel:zeroGradParameters"></a>
 * `fbcunn.AbstractParallel:zeroGradParameters()`
<a name="fbcunn.AbstractParallel:updateParameters"></a>
 * `fbcunn.AbstractParallel:updateParameters(learningRate)`
<a name="fbcunn.AbstractParallel:share"></a>
 * `fbcunn.AbstractParallel:share(mlp,...)`
<a name="fbcunn.AbstractParallel:clone"></a>
 * `fbcunn.AbstractParallel:clone()`
<a name="fbcunn.AbstractParallel:reset"></a>
 * `fbcunn.AbstractParallel:reset(stdv)`
