--[[
Gaussian dataset
By Xiang Zhang (xiang.zhang [at] nyu.edu) @ New York University
Version 0.1, 02/01/2013

This file is implemented for the course CSCI-GA.3033-002 Big Data: Large Scale
Machine Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu) and professor John Langford (langford [at] cs.nyu.edu)

This script generates datasets sampled from several gaussian distributions. To
use it, simply call gaussdata:getDatasets. E.g.
t7> train_data, test_data = gaussdata:getDatasets(3000,1000)
The code above will generate a training dataset of 3000 samples and a testing
dataset of 1000 samples from two gaussian distributions centered at [-1,-1]
and [1,1] with their covariance as identity matrices. The labels of these two
classes are [-1,1].

You can customize the datasets with arbitrary dimension and for arbitrary
number of gaussian stumps, just by defining their centers, covariances and
labels. For example,
-- Generate centers of two gaussians at (3,3) and (-3,-3)
t7> centers = {torch.Tensor({3,3}), torch.Tensor({-3,-3})}
-- Using 2 times the identity matrix as covariance for the first gaussian, 3
-- for the second.
t7> covs = {torch.eye(2)*2, torch.eye(2)*3}
-- The label of the first gaussian is -1, the second is 1
t7> labels = {-1,1}
-- Generate dataset of 3000 trainig samples and 1000 testing samples
t7> train_data,test_data = gaussdata:getDatasets(3000,1000,centers,covs,labels)
]]

-- Required packages
require("torch")
require("math")

-- The namespace
gaussdata = gaussdata or {}

-- Generate datasets of gaussian bumps
-- centers: the table of centers of each gaussian. The dimension is read as the size of the first center
-- covariances: the table of covariances of each gaussian. Must match with the size of centers and dimension.
-- labels: the table of labels for each gaussian.
function gaussdata:generate(size, centers, covariances, labels)
   -- Compute matrix square-root on each of the covariances
   local covsqrt = {}
   -- Iterate through all the covariances
   for i = 1,#covariances do
      -- Do symmetrical eigen-decomposition
      local e,v = torch.symeig(covariances[i],"V")
      -- Compute the covariance
      covsqrt[i] = v:t()*torch.diag(torch.sqrt(e))*v
   end
   -- Create the dataset
   local dataset = {}
   -- Define the size of the dataset
   function dataset:size() return size end
   -- Define the features (inputs) of the dataset
   function dataset:features() return centers[1]:size(1) end
   -- Define the classes of the dataset
   function dataset:classes() return #centers end
   -- Generate samples of data
   for i = 1, size do
      -- Determine the label
      local class = math.random(#centers)
      -- Get the inputs
      local input = centers[class] + covsqrt[class]*torch.randn(centers[class]:size(1))
      -- Get the outputs
      local output = labels[class]
      -- Insert it into dataset
      dataset[i] = {input, output}
   end
   -- Return the dataset
   return dataset
end

-- Get gaussian datasets
-- train_size: size of the training dataset
-- test_size: size of the testing dataset
-- centers: a table of centers for the gaussians
-- covariances: a table of covariances for the gaussians
-- labels: a table of labels for different gaussians
function gaussdata:getDatasets(train_size, test_size, centers, covariances, labels)
   -- Default centers are [-1,-1] and [1,1]
   centers = centers or {torch.Tensor({-1,-1}),torch.Tensor({1,1})}
   -- Default covariances are identity matrices
   covariances = covariances or {}
   -- Loop over all the centers
   for i = 1,#centers do
      -- Make default center the identity matrix
      covariances[i] = covariances[i] or torch.eye(centers[i]:size(1))
   end
   -- Default labels are -1,1 for 2 classes, or class index for multiple classes
   if labels == nil and #centers == 2 then
      -- 2-class labels
      labels = {-1,1}
   else
      -- Multi-class labels
      for i = 1,#centers do
	 -- Set it to the class number
	 labels[i] = labels[i] or i
      end
   end
   -- Generate the training data
   train = gaussdata:generate(train_size, centers, covariances, labels)
   -- Generate the testing data
   test = gaussdata:generate(test_size, centers, covariances, labels)
   -- Return the datasets
   return train, test
end