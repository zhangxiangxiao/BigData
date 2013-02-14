--[[
xtools package
By Xiang Zhang @ New York University
Version 0.1, 02/04/2013
]]

-- Required package
require("torch")

-- The xtools namespace
xtools = xtools or {}

-- Fix the bug in torch.packageLuaPath
function torch.packageLuaPath(name)
   if not name then
      local ret = string.match(torch.packageLuaPath('torch'), '(.*)/')
       if not ret then --windows?
	  ret = string.match(torch.packageLuaPath('torch'), '(.*)\\')
       end
       return ret 
   end
   for path in string.gmatch(package.path, "[^;]+") do
      path = string.gsub(path, "%?", name)
      local f = io.open(path)
      if f then
         f:close()
         local ret = string.match(path, "(.*)/")
         if not ret then --windows?
	    ret = string.match(path, "(.*)\\")
         end
         return ret
      end
   end
end

-- Add a path
function xtools.addpath(path)
   local exist = false
   for p in string.gmatch(package.path, "[^;]+") do
      if p == path then
	 exist = true
      end
   end
   if exist == false then
      package.path = package.path..";"..path
   end
end