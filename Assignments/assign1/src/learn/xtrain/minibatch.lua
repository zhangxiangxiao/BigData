--[[
Trainer implementation
By Xiang Zhang @ New York University
Version 0.1, 02/04/2013
]]

--Required packages
require("nn")
require("optim")
require("xlua")

-- The namespace
xtrain = xtrain or {}

-- A stochastic trainer (optimize once for each sample) using any algorithm from optim package
local minibatch = torch.class("xtrain.minibatch")

-- Initializing the trainer
-- model: Some mod constructed using nn package (no criterion included)
-- loss: A loss following the criterion convention in nn package
-- regu: (optional) A regularization object.
-- decfunc: (optional) a function returns the decision when passing in the output of the model. Default is return the output itself.
-- errfunc: (optional) A function returns the per-sample error when passing in the predicted decision and the correct label. Default is equality comparison.
-- optalg: (optional) An algorithm follows the convention of the optim package. Default is optim.sgd
-- state: (optional) The state to be put into the optalg. Refer to the optim package for more info.
-- config: (optional) The configuration of the trainer. A table containing values
function minibatch:__init(model, loss, regu, decfunc, errfunc, optalg, state, config)
   self.model = model
   self.loss = loss
   self.regu = regu
   self.decfunc = decfunc or function (output) return output end
   self.errfunc = errorfunc or function (decision,label) return decision == label and 0 or 1 end
   self.optalg = optalg or optim.sgd
   self.state = state or {}
   self.config = config or {}

   -- By default show the progress
   if self.config.showProgress == nil then self.config.showProgress = true end
   -- Default batch size is 20
   self.config.batchSize = self.config.batchSize or 20

   -- The default regularizer is no regularization
   if self.regu == nil then
      self.regu = {}
      function self.regu:forward(input) return 0 end
      function self.regu:backward(input) return 0 end
   end
end

-- Train on a dataset
-- dataset: a dataset follows the torch dataset convention.
-- epoch: maximum number of passes over the entire training data.
-- The return value is the result of stochastic:test(dataset) after training.
function minibatch:train(dataset, epoch)
   -- Retrieve parameters and gradients
   local w, dw = self.model:getParameters()
   -- Iterate over epoches
   for t = 1, epoch do
      -- Display progress
      if self.config.showProgress == true then xlua.progress(t, epoch) end
      -- Define the closure to evalute l and dl
      local function feval(x)
	 -- get new parameters
	 if x ~= w then w:copy(x) end
	 -- Reset gradients and losses
	 local loss = 0
	 dw:zero()
	 -- Iterate over batchSize samples
	 for i = 1, self.config.batchSize do
	    -- Get the index
	    local index = math.mod((t-1)*self.config.batchSize+i-1, dataset:size()) + 1
	    -- Forward propagation on the model
	    local output = self.model:forward(dataset[index][1])
	    -- Forward propagation on the loss
	    loss = loss + self.loss:forward(output, dataset[index][2])
	    -- Backward propagation on the loss
	    local dout = self.loss:backward(output, dataset[index][2])
	    -- Backward propagation on the model
	    local dx = self.model:backward(dataset[index][1], dout)
	 end
	 -- Get the regularization loss
	 local r = self.regu:forward(w)
	 -- Get the regularization gradient
	 local dr = self.regu:backward(w)
	 -- Return loss and gradients
	 return (loss+r)/self.config.batchSize, dw:add(dr):div(self.config.batchSize)
      end
      -- Optimize current iteration
      self.optalg(feval, w, self.state)
   end
   error, loss = self:test(dataset)
   -- Return testing results
   return error,loss
end

-- Test on a dataset. Return value is error, loss pair.
-- dataset: a dataset follows the torch dataset convention.
function minibatch:test(dataset)
   -- Initialize error variable
   local err = 0
   -- Initialize loss variable
   local loss = 0
   -- Get parameters
   local w, dw = self.model:getParameters()
   -- Get regularization loss
   local r = self.regu:forward(w)
   -- Iterate over all the training data
   for i = 1,dataset:size() do
      -- Forward propagation on the sample
      local output = self.model:forward(dataset[i][1])
      -- Get the decision of the sample
      local decision = self.decfunc(output)
      -- Get the error of the sample. Iterative averaging.
      err = err/i*(i-1) + self.errfunc(decision, dataset[i][2])/i
      -- Get the loss of the sample. Iterative averaging.
      loss = loss/i*(i-1) + self.loss:forward(output, dataset[i][2])/i
   end
   -- Add the regularization
   loss = loss + r/self.config.batchSize
   -- Return the values
   return err, loss
end
