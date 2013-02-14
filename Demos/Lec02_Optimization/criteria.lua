--[[
Extra criteria for the demo
By Xiang Zhang @ New York University
Version 0.1, 02/05/2013
]]

local MSECriterion, parent = torch.class('nn.MMSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:updateOutput(input, target)
   local t = torch.Tensor{target}
   return input.nn.MSECriterion_updateOutput(self, input, t)
end

function MSECriterion:updateGradInput(input, target)
   local t = torch.Tensor{target}
   return input.nn.MSECriterion_updateGradInput(self, input, t)
end