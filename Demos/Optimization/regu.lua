--[[
Regularizers implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) @ New York University
Version 0.1, 02/04/2013
]]

-- No regularization
function reguNone()
   local regularizer = {}
   -- The loss is 0
   function regularizer:forward(input) return 0 end
   -- The gradient is 0
   function regularizer:backward(input) return 0 end
   -- Return this regularizer
   return regularizer
end

-- l2 regularization module
-- lambda: the regularization parameter
function reguL2(lambda)
   local regularizer = {}
   -- The loss value of this regularizer
   function regularizer:forward(input) return torch.sum(torch.pow(input,2))*lambda end
   -- The gradient of l with respect to w
   function regularizer:backward(input)
      self.dr = self.dr or torch.Tensor()
      self.dr:resizeAs(input):copy(input)
      self.dr:mul(2*lambda)
      return self.dr
   end
   -- Return this regularizer
   return regularizer
end

-- l1 regularization module
-- lambda: the regularization parameter
function reguL1(lambda)
   local regularizer = {}
   -- The loss value of this regularizer
   function regularizer:forward(input) return torch.sum(torch.abs(input))*lambda end
   -- The gradient of l with respect to w
   function regularizer:backward(input)
      self.dr = self.dr or torch.Tensor()
      self.dr:resizeAs(input):copy(torch.sign(input))
      self.dr:mul(lambda)
      return self.dr
   end
   -- Return this regularizer
   return regularizer
end
