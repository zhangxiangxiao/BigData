--[[
Quasi-Newton methods and SGD demo
By Xiang Zhang @ New York University
Version 0.1, 02/04/2013

DOCUMENTATION

These demos require the following packages:
xlua, optim
They can be installed using the torch-pkg command

To try different models, please edit the 'model' variable. The model could be
anything following the nn.Module protocol.

To try different loss functions, please edit the 'loss' variable. The loss
could be anything following the nn.Criterion protocol.

To try different training mechanisms, please edit the 'trainer' variable. We
have three simple trainers: xtrain.stochastic, xtrain.batch, xtrain.minibatch.
It is recommended to use xtrain.batch or xtrain.minibatch if quasi-Newton
method (such as bfgs, lbfgs, cg) is used.

To try different learning algorithms, please edit the 'optalg' variable. You
can change it to optim.sgd, optim.cg, optim.bfgs, optim.lbfgs, etc.

To try different regularization methods, please edit the 'regu' variable. You
can change it to reguL2(lambda), reguL1(lambda) or reguNone(). 'lambda' is the
regularization parameter. regu == nil is equivalent to regu == reguNone().

Other changes (such as how the training samples are drawn and the size of a
minibatch) could be made by finding the appropriate variables. Please take
your time to read the code a bit.
]]


-- Load global libraries
require("nn")
require("optim")
require("gnuplot")
require("qt")
require("qttorch")
require("qtwidget")
require("qtuiloader")
require("image")

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

-- Load extra criteria
if nn.MMSECriterion == nil then dofile("criteria.lua") end

-- Configuration of datasets
local train_size = 500
local test_size = 100
local centers = {torch.Tensor({2,1}), torch.Tensor({-1,-2})}
local covs = {torch.eye(2)*10, torch.eye(2)*7}
local labels = {-1,1}

-- Configuration of model
local model = nn.Linear(2,1)

-- Configuration of loss
--local loss = nn.MarginCriterion(1)
local loss = nn.MMSECriterion()

-- Configuration of regularizer
local regu = reguL2(0.05)

-- Configuration of optimizer
local state = {}
-- If you want to use different training ratio for the weights
-- state.learningRates = torch.Tensor{1/2,1,1}
state.learningRate = 0.002
-- Optimization algorithm used is optim.sgd
local optalg = optim.sgd


-- Configuration of trainer
local config = {batchSize = 1}
-- Decision function is sign for binary linear classification
local decfunc = function(output) return torch.sign(output)[1] end
-- Error is equality comparison
local errfunc = function(decision, label) return decision == label and 0 or 1 end
-- Use a minibatch trainer
local trainer = xtrain.minibatch(model, loss, regu, decfunc, errfunc, optalg, state, config)

-- Configuration of training
local epoches = 2*500/config.batchSize

-- Start training
-- Set the parameter initial values
local params, grads = model:getParameters()
params[1] = -0.5
params[2] = 0.5
local data_train, data_test = gaussdata:getDatasets(train_size, test_size, centers, convs, labels)
-- Train for epoches steps
local error_train, loss_train, wtable = trainer:train(data_train, epoches)
-- Test on testing data
local error_test, loss_test = trainer:test(data_test)

-- Report training results
print("error_train = "..error_train)
print("error_test = "..error_test)
print("loss_train = "..loss_train)
print("loss_test = "..loss_test)

-- Plotting configurations
local xrange = {-1,1}
local yrange = {-1,1}
local gridSize = 25

-- This function generates energy for one sample
local enefunc = function(input)
   -- Get the parameters of the model
   local w, dw = model:getParameters()
   -- Modify the weight parameters
   w[1] = input[1]
   w[2] = input[2]
   -- Get the training loss
   local error, energy = trainer:test(data_train)
   -- Return the energy
   return energy
end

-- This function generates grid of inputs and energy
local gridfunc = function(xgrid, ygrid)
   -- Do meshgrid
   local x = torch.zeros(ygrid:size(1), xgrid:size(1))
   local y = torch.zeros(ygrid:size(1), xgrid:size(1))
   for i = 1,ygrid:size(1) do
      x[{i,{}}]:copy(xgrid)
   end
   for i = 1,xgrid:size(1) do
      y[{{},i}]:copy(ygrid)
   end
   -- Compute the energies
   local z = torch.zeros(ygrid:size(1), xgrid:size(1))
   for i = 1,z:size(1) do
      xlua.progress(i, z:size(1))
      for j = 1,z:size(2) do
	 local input = torch.Tensor{x[{i,j}],y[{i,j}]}
	 z[{i,j}] = enefunc(input)
      end
   end
   -- Return the values
   return x,y,z
end

-- Generate grid and plot
gnuplot.figure(1)
local xgrid = torch.linspace(xrange[1],xrange[2],gridSize)
local ygrid = torch.linspace(yrange[1],yrange[2],gridSize)
local x,y,z = gridfunc(xgrid,ygrid)
gnuplot.splot(x,y,z)

-- Generate the energy associated with the weights
local plotfunc = function (wtable)
   -- Build x, and y values as tables of tensors
   local x = torch.Tensor(#wtable)
   local y = torch.Tensor(#wtable)
   local z = torch.Tensor(#wtable)
   -- Iterate over all the values in wtable
   for i = 1, #wtable do
      xlua.progress(i,#wtable)
      x[i] = wtable[i][1]
      y[i] = wtable[i][2]
      z[i] = enefunc(wtable[i])
   end
   return x,y,z
end

-- Get the tracks of variables
local xtrack, ytrack,ztrack = plotfunc(wtable)
-- Plot these tracks
gnuplot.figure(2)
gnuplot.plot(xtrack,ytrack,"+-")
gnuplot.figure(3)
gnuplot.plot(ztrack,"+-")

-- Mark the values on the energy surface
local zene = (z-z:min())/(z:max()-z:min())
zene:pow(0.2)
-- Draw using Qt widget
local width = 600
local height = 600
local colors = {"red","blue"}
local w = qtwidget.newwindow(width,height,"Energy surface")
local qimg = qt.QImage.fromTensor(image.scale(zene,width,height))
w:image(0,0,qimg)
w:stroke()
local xindex = (wtable[1][1]-xrange[1])/(xrange[2]-xrange[1])*width
local yindex = (wtable[1][2]-yrange[1])/(yrange[2]-yrange[1])*height
w:setlinewidth(5)
local colorvar = 1
w:setcolor(colors[math.mod(colorvar,2)+1])
for i = 2,#wtable do
   if i*config.batchSize >= colorvar*data_train:size() then
      colorvar = colorvar + 1
      w:setcolor(colors[math.mod(colorvar,2)+1])
   end
   w:moveto(xindex,yindex)
   xindex = (wtable[i][1]-xrange[1])/(xrange[2]-xrange[1])*width
   yindex = (wtable[i][2]-yrange[1])/(yrange[2]-yrange[1])*height
   w:lineto(xindex,yindex)
   w:stroke()
end
