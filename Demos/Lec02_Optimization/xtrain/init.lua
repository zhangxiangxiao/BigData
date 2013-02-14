--[[
Initializer for the trainer package
By Xiang Zhang @ New York University
Version 0.1, 01/26/2013
]]

-- Required packages
require "torch"

-- The namespace
xtrain = xtrain or {}

-- trainers
torch.include("xtrain","stochastic.lua")
torch.include("xtrain","batch.lua")
torch.include("xtrain","minibatch.lua")