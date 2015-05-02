

### ModelParallel.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbcunn.ModelParallel.dok"></a>


## fbcunn.ModelParallel ##

 `ModelParallel` copies inputs to all child modules, and runs
disjoint parts of the model on separate devices.

For example, consider a convolutional layer with a large number of
filter banks. ModelParallel will split the model along the given
`dimension` (e.g. 2 if we lay the input out as `BDWH`), copy the input
to each device, and then merge the outputs across the device.



#### Undocumented methods ####

<a name="fbcunn.ModelParallel"></a>
 * `fbcunn.ModelParallel(dimension)`
<a name="fbcunn.ModelParallel:nextGPU"></a>
 * `fbcunn.ModelParallel:nextGPU()`
<a name="fbcunn.ModelParallel:add"></a>
 * `fbcunn.ModelParallel:add(module, gpuid)`
<a name="fbcunn.ModelParallel:get"></a>
 * `fbcunn.ModelParallel:get(index)`
<a name="fbcunn.ModelParallel:name"></a>
 * `fbcunn.ModelParallel:name()`
<a name="fbcunn.ModelParallel:updateGradInput"></a>
 * `fbcunn.ModelParallel:updateGradInput(_input, gradOutput)`
