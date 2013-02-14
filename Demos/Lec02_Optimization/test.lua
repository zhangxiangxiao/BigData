--[[ Test the xtrain package
By Xiang Zhang @ New York University
Version 0.1, 02/04/2013
]]


-- Load global libraries
require("nn")
require("optim")

-- Load the xtools library. Fixing bugs of torch and etc.
dofile("xtools.lua")
-- Add a path to support local libraries
xtools.addpath("./?/init.lua")
-- Load local libraries
require("xtrain")

-- Load dataset
dofile("gaussdata.lua")

-- Load regularizers
dofile("regu.lua")

-- Configuration of datasets
local train_size = 3000
local test_size = 1000
local centers = {torch.Tensor({3,3}), torch.Tensor({-3,-3})}
local covs = {torch.eye(2)*2, torch.eye(2)*3}
local labels = {-1,1}

-- Configuration of model
local model = nn.Linear(2,1)

-- Configuration of loss
local loss = nn.MarginCriterion(1)

-- Configuration of regularizer
local regu = reguL2(0.05)

-- Configuration of optimizer
local state = {}
-- Optimization algorithm used is optim.sgd
local optalg = optim.bfgs


-- Configuration of trainer
local config = {batchSize = 50}
-- Decision function is sign for binary linear classification
local decfunc = function(output) return torch.sign(output)[1] end
-- Error is equality comparison
local errfunc = function(decision, label) return decision == label and 0 or 1 end
-- Use a minibatch trainer
local trainer = xtrain.minibatch(model, loss, regu, decfunc, errfunc, optalg, state, config)

-- Configuration of training
local epoches = 100

-- Start training
local data_train, data_test = gaussdata:getDatasets(train_size, test_size, centers, convs, labels)
-- Train for epoches steps
local error_train, loss_train = trainer:train(data_train, epoches)
-- Test on testing data
local error_test, loss_test = trainer:test(data_test)

-- Report training results
print("error_train = "..error_train)
print("error_test = "..error_test)
print("loss_train = "..loss_train)
print("loss_test = "..loss_test)

-- The function to generate energy
local enefunc = function(inputs)
	-- Forward propagation on the sample
      local output = self.model:forward(dataset[i][1])
end
