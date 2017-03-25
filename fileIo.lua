require 'Rect'
require 'image'

function load_image_auto_size(fn, target_smaller_side, max_pixel_size, color_space)
    local img = image.load(fn, 3, 'float')
    local dim = img:size()
    local w, h
    if dim[2] < dim[3] then
      -- height is smaller than width, set h to target_size
      w = math.min(dim[3] * target_smaller_side/dim[2], max_pixel_size)
      h = dim[2] * w/dim[3]
    else
      -- width is smaller than height, set w to target_size
      h = math.min(dim[2] * target_smaller_side/dim[1], max_pixel_size)
      w = dim[3] * h/dim[2]
    end
    img = image.scale(img, w, h)
    if color_space == 'yuv' then
      img = image.rgb2yuv(img)
    elseif color_space == 'lab' then
      img = image.rgb2lab(img)
    elseif color_space == 'hsv' then
      img = image.rgb2hsv(img)
    end
    return img, dim
  end
  

function save_obj(file_name, obj)
  local f = torch.DiskFile(file_name, 'w')
  f:writeObject(obj)
  f:close()
end

function load_obj(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local obj = f:readObject()
  f:close()
  return obj
end

function save_model(file_name, weights, options, stats)
  save_obj(file_name,
    {
      version = 0,
      weights = weights,
      options = options,
      stats = stats
    })
end

function load_model(cfg, model_path, network_filename, cuda)
  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)
  graph.dot(model.feature_net.fg, 'feature_net',string.format('%s/feature_net',opt.resultDir))
  graph.dot(model.feature_net.bg, 'feature_net',string.format('%s/feature_net',opt.resultDir))
  graph.dot(model.DQN_net.fg, 'DQN_net', string.format('%s/DQN_net',opt.resultDir))
  graph.dot(model.DQN_net.bg, 'DQN_net', string.format('%s/DQN_net',opt.resultDir))

  if cuda then
    model.feature_net:cuda()
    model.DQN_net:cuda()
  end
  local weights = {}
  local gradient
  -- combine parameters from pnet and cnet into flat tensors
  weights[1], gradient = combine_and_flatten_parameters(model.feature_net,model.DQN_net)
  local training_stats
  if network_filename and #network_filename > 0 then
    local stored = load_obj(network_filename)
    training_stats = stored.stats
    weights[1]:copy(stored.weights)
  end
  return model, weights[1], gradient, training_stats
end

function combine_and_flatten_parameters(...)
  local nets = { ... }
  local parameters,gradParameters = {}, {}
  for i=1,#nets do
    local w, g = nets[i]:parameters()
    for i=1,#w do
      table.insert(parameters, w[i])
      table.insert(gradParameters, g[i])
    end
  end
  return nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
end