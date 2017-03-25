function train(model,X_train,y_train,batch_size,epoch,optimState)
  model:training()
  epoch = epoch or 1
  parameters,gradParameters = model:getParameters()
  criterion = nn.CrossEntropyCriterion()
  -- drop learning rate every "epoch_step" epochs
  --if epoch % epoch == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. batch_size .. ']')

  local targets = torch.FloatTensor(batch_size)
  local indices = torch.randperm(X_train:size(1)):long():split(batch_size)
  -- remove last element so that all the batches have equal size
  --indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = X_train:index(1,v)
    --print('v',v:size())
   -- print('input_size',inputs:size())
    
    --print('y_trian_size',y_train:size())
    targets:copy(y_train:index(1,v))
    --print('y_train[1]',y_train[1])
    --print('X_train[1]',X_train[1])
    --print('targets[1]',targets[1])
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      print('inputs size',inputs:size())
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      
      print('outputs_size',outputs[1])
      print('target_size',targets[1])
      --print('outputs',outputs)
      --print('targets',targets)
      confusion:batchAdd(outputs, targets)
      
      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: %.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


--test
require 'nn'
require 'optim'
require 'torch'
net = nn.Sequential()
net:add(nn.Linear(100,36))
net:add(nn.ReLU(true))
net:add(nn.Linear(36,6))
net:add(nn.LogSoftMax())

local optimState = {
    learningRate = 0.1,
    weightDecay = 1E-5,
    momentum = 0.8,
    nesterov = true,
    learningRateDecay = 0,
    dampening =0.0
}

local input_x = torch.randn(600,100)
local input_y = torch.Tensor(600):apply(function() return torch.random(6) end)
local batch_size =100
local epoch=100

confusion = optim.ConfusionMatrix(6)
for i=1,epoch do
    train(net,input_x,input_y,batch_size,epoch,optimState)
end