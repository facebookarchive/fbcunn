-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
Group k-max pooling performs pooling along a dimension of arbitrary length
(e.g. a sentence) down to a length of ${k}$.

Given a matrix where rows are words and columns are embedding dimensions, we
compute the ${L^2}$ norm of each word:

```
   o---------o
w1 |         | -> norm1
w2 |         | -> norm2
w3 |         | -> norm3
w4 |         | -> norm4
   o---------o
```

Group K-max pooling keeps the K words with largest norm and discards the
rest.
]]   
local GroupKMaxPooling, parent =
    torch.class('nn.GroupKMaxPooling', 'nn.Module')

function GroupKMaxPooling:__init(k, k_dynamic)
    parent.__init(self)

    self.k = k
    self.k_dynamic = k_dynamic or -1

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.switches = torch.LongTensor()
end

function GroupKMaxPooling:updateOutput(input)
    if input:dim() == 2 then
        group_norms = torch.norm(input, 2, 2)
    else
        group_norms = torch.norm(input, 2, 3)
    end

    input = input:contiguous()

    return input.nn.GroupKMaxPooling_updateOutput(self, input, group_norms)
end

function GroupKMaxPooling:updateGradInput(input, gradOutput)
    input = input:contiguous()
    gradOutput = gradOutput:contiguous()

    return input.nn.GroupKMaxPooling_updateGradInput(self, input, gradOutput)
end
