require 'nn'
require 'fbnn'
require 'cunn'
require 'libfbcunn'
require 'libfbcunnlayers'

include('AbstractParallel.lua')
include('CuBLASWrapper.lua')
include('DataParallel.lua')
include('FeatureLPPooling.lua')
include('FFTWrapper.lua')
-- include('HalfPrecision.lua')
include('LookupTableGPU.lua')
include('ModelParallel.lua')
include('OneBitDataParallel.lua')
include('OneBitQuantization.lua')
include('OneBitSGD.lua')
include('SpatialConvolutionCuFFT.lua')
include('TemporalConvolutionFB.lua')
include('TemporalKMaxPooling.lua')

-- Monkey-patch module to include getParametersByDevice
-- Get the params of the module separated by device.
-- Returns the pair:
--   {0 = flat tensor containing CPU weights,
--    1 = flat tensor containing weights from device 1,
--    ...
--    N = ... containing weights from device N},
--   {0 = flat tensor containing CPU grads,
--    1 = ... containing grads from device 1, ...}
function nn.Module:getParametersByDevice()
    local n_dev = cutorch.getDeviceCount()
    local d2weights = {} -- Device => { tensor1, tensor2, ..., tensorN }
    local d2grads   = {} -- Device => { tensor1, tensor2, ..., tensorN }

    local function tensor_to_dev(tensor)
        local tnm = torch.typename(tensor)
        if tnm == 'torch.CudaTensor' then
            return tensor:getDevice()
        end
        return 0
    end

    local params, grads = self:parameters()
    assert(#params == #grads)
    -- Herd each tensor into appropriate row of weights,grads
    for i = 1,#params do
        local p = params[i]
        local g = grads[i]
        local d = tensor_to_dev(p)
        if d ~= tensor_to_dev(g) then
            error(("Improbable module; params,grads on devices %d,%d"):
                  format(d, tensor_to_dev(g)))
        end
        if not d2weights[d] then
            d2weights[d] = {}
            d2grads[d] = {}
        end
        table.insert(d2weights[d], p)
        table.insert(d2grads[d], g)
    end

    local function gather(dev, params, grads)
        if not params or #params == 0 then
            return nil
        end
        if dev == 0 then
            return nn.Module._gather(params), nn.Module._gather(grads)
        end
        return cutorch.withDevice(dev,
            function() return nn.Module._gather(params),
                              nn.Module._gather(grads)
        end)
    end

    local ret_params = { }
    local ret_grads = { }
    for d = 0,n_dev do -- sic
        ret_params[d], ret_grads[d] = gather(d, d2weights[d], d2grads[d])
    end

    return ret_params, ret_grads
end
