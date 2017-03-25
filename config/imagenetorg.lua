 local imgnet_cfg = {
  action_count = 6,  -- excluding background class
  epsilon = 1,
  target_smaller_side = 400,
  uniform_image_scaling = true,
  scales = { 64, 128, 256 },
  max_pixel_size = 1000,
  -- normalization = { scaling = true, centering = true, method = 'contrastive', width = 7 },
  normalization = { },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'rgb',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  history_count= 4 ,
  experience_momery = 2000,
  background_base_path = '',
  batch_size = 64,
  positive_threshold = 0.7,
  negative_threshold = 0.3,
  best_match = true,
  nearby_aversion = false,
  backgroundClass = 2
}

return imgnet_cfg
