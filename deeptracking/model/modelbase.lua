--
-- User: mathieu
-- Date: 19/12/16
-- Time: 8:34 AM
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'math'

local ModelBase = torch.class('ModelBase')

function ModelBase:__init(backend, optimfunc)
    self.net = nil
    self.backend = backend
    self.optimFunction = optim.adam
    if optimfunc == "sgd" then
       self.optimFunction = optim.sgd
    elseif optimfunc == "adadelta" then
        self.optimFunction = optim.adadelta
    end
    self.config = {
        learningRate = 0.005,
        learningRateDecay = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-08,
        momentum = 0.9,
        dampening = 0,
        nesterov = 0.9,
    }
end

function ModelBase:set_configs(dict)
    for k, v in pairs(dict) do
        self.config[k] = v
    end
end

function ModelBase:get_configs(name)
    return self.config[name]
end

function ModelBase:show_model()
    print(string.format("Backend : %s", self.backend))
    print(self.net)
end

function ModelBase:show_memory_info()
    require 'cutorch'
    idx = cutorch.getDevice()
    freeMemory, totalMemory = cutorch.getMemoryUsage(idx)
    print(string.format("Free memory : %f Gb, Total memory : %f Gb", freeMemory/1073741824.0, totalMemory/1073741824.0))
end

-- Convert tensor based on backend requested
function ModelBase:setup_tensor(ref, buffer)
    local localOutput = buffer
    if self.backend == 'cpu' then
        localOutput = ref
    else
        localOutput = localOutput or ref:clone()
        if torch.type(localOutput) ~= 'torch.CudaTensor' then
            localOutput = localOutput:cuda()
        end
        localOutput:resize(ref:size())
        localOutput:copy(ref)
    end
    return localOutput
end

function ModelBase:set_backend(module)
    if self.backend == 'cuda' then
        module = module:cuda()
    else
        module = module:float()
    end
    return module
end

function ModelBase:convert_inputs(inputs)
    -- this function is used when you have particular inputs, it handles backend transfer and any formating to the input data
    error("convert_inputs not defined!")
end

function ModelBase:convert_outputs(outputs)
    -- convert forward outputs so it can be handled in python
    error("convert_outputs not defined!")
end

function ModelBase:compute_criterion(forward_input, label)
    -- compute the criterion given the output of forward and labels, returns a dict with losses :
    -- label : the generic loss used for trainning algorithm
    -- user_defined_loss : any other loss.
    error("compute_criterion not defined!")
end

function ModelBase:extract_features()
    -- This function return a dict containning layers activations. By default it will return nil
    return nil
end

function ModelBase:extract_grad_statistic()
    -- This function return a dict containning gradient information after backward pass.
    return nil
end

function ModelBase:on_train()
    -- Will be called when train is called. can be reimplemented by subclasses
end

function ModelBase:init_model()
    self.net = self:set_backend(self.net)
    self.params, self.gradParams = self.net:getParameters()
end

function ModelBase:train(inputs, labels)
    self.net:training()
    self:on_train()
    local func = function(x)
        collectgarbage()
        self.gradParams:zero()
        local converted_inputs = self:convert_inputs(inputs)
        local output = self.net:forward(converted_inputs)
        losses, f_grad = self:compute_criterion(output, labels)
        self.net:backward(converted_inputs, f_grad)
       return losses['label'], self.gradParams
    end
    self.optimFunction(func, self.params, self.config)
    return losses
end

function ModelBase:test(inputs)
    collectgarbage(); collectgarbage()
    self.net:evaluate()
    local converted_inputs = self:convert_inputs(inputs)
    local output = self.net:forward(converted_inputs)
    -- Pytorch does not support gpu tensor output
    return self:convert_outputs(output, "cpu")
end

function ModelBase:loss_function(prediction, truth)
    local prediction_b = self:convert_outputs(prediction, self.backend)
    self.truthTensor = self:setup_tensor(truth, self.truthTensor)
    losses, f_grad = self:compute_criterion(prediction_b, self.truthTensor)
    return losses
end

function ModelBase:save(path, light)
    if light then
        self.net:clearState()
    end
    torch.save(path..".t7", self.net)
    torch.save(path.."_optim.t7", self.config)
end

function ModelBase:load(path)
    self.net = torch.load(path..".t7")
    self.config = torch.load(path.."_optim.t7")
    self:init_model()
end

