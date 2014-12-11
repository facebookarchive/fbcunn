-- Copyright 2004-present Facebook. All Rights Reserved.

local KMaxPooling, parent =
    torch.class('nn.KMaxPooling', 'nn.Module')

function KMaxPooling:__init(k, k_dynamic)
    parent.__init(self)

    self.k = k
    self.k_dynamic = k_dynamic or -1

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.switches = torch.LongTensor()
end

function KMaxPooling:updateOutput(input, input_info)
    input = input:contiguous()

    local return_info = true

    if input_info == nil then
        input_length = torch.LongTensor(1)
        input_length[1] = input:size(1)

        input_info = { length = input_length }

        return_info = false
    end

    self.input_length = input_info.length:clone()
    -- updated in place
    self.output_length = input_info.length:clone()

    input.nn.KMaxPooling_updateOutput(self, input)

    if return_info then
        return {
            output = self.output,
            info = { length = self.output_length },
        }
    else
        return self.output
    end
end

function KMaxPooling:updateGradInput(input, gradOutput)
    input = input:contiguous()
    gradOutput = gradOutput:contiguous()

    return input.nn.KMaxPooling_updateGradInput(self, input, gradOutput)
end
