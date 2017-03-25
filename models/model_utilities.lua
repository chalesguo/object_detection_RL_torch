require 'nngraph'
--require '../models/vggorg_small'

function create_conv_layers(layers, input)
  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, bn)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, 1,1, padW,padH))
    if bn then
      container:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    end
    container:add(nn.ReLU(true))
    if dropout and dropout > 0 then
      container:add(nn.SpatialDropout(dropout))
    end
    return container
  end

  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps, pooling)
    local bn = false
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, bn)
      nInputPlane = nOutputPlane
      dropout = nil -- only one dropout layer per conv-pool block
      bn = false
    end
    if pooling == 'max' then
      container:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
    end
    return container
  end

  local conv_outputs = {}

  local inputs = 3
  local prev = input
  for i,l in ipairs(layers) do
    local net = nn.Sequential()
    ConvPoolBlock(net, inputs, l.filters, l.kW, l.kH, l.padW, l.padH, l.dropout, l.conv_steps, l.pooling)
    inputs = l.filters
    prev = net(prev)
    if i >3 then
      table.insert(conv_outputs, prev)
    end
  end

  return conv_outputs
end

function gaussian_init(module, name)
  local function init_module(m)
    for k,v in pairs(m:findModules(name)) do
      local n = v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2 / n))
      v.bias:zero()
    end
  end
  module:apply(init_module)
end

function create_feature_net(layers)

  -- creates an anchor network which reduces the input first to 256 dimensions
  -- and then further to the anchor outputs for 3 aspect ratios
  local input = nn.Identity()()
  local conv_outputs = create_conv_layers(layers, input)

    -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
  local model = nn.gModule({ input }, conv_outputs)
  gaussian_init(model, 'nn.SpatialConvolution')
  return model
end

function create_DQN_net(inputs, action_count, class_layers)
  -- create classifiaction network
  local net = nn.Sequential()

  local prev_input_count = inputs
  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_count, l.n))
    if l.batch_norm then
      net:add(nn.BatchNormalization(l.n))
    end
    net:add(nn.PReLU())
    if l.dropout and l.dropout > 0 then
      net:add(nn.Dropout(l.dropout))
    end
    prev_input_count = l.n
  end

  local input = nn.Identity()()
  local node = net(input)

  -- now the network splits into regression and classification branches

  -- regression output
  --local rout = nn.Linear(prev_input_count, 4)(node)

  -- classification output
  local cnet = nn.Sequential()
  --print('prev_input_count',prev_input_count)
  --print('class_count',class_count)
  cnet:add(nn.Linear(prev_input_count, action_count))
  --cnet:add(nn.LogSoftMax())
  local cout = cnet(node)

  -- create bbox finetuning + classification output
  local model = nn.gModule({ input }, { cout })

  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.SpatialConvolution')

  return model
end

function create_model(cfg, layers, class_layers)
  local DQN_net_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * layers[#layers].filters+cfg.action_count*cfg.history_count
  local model =
  {
    cfg = cfg,
    layers = layers,
    feature_net = create_feature_net(layers),
    DQN_net = create_DQN_net(DQN_net_ninputs, cfg.action_count, class_layers)
  }
  return model
end



----------------
--test