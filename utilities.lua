require 'lfs' -- lua file system for directory listings
require 'nn'
local image = require 'image'
require 'Rect'
function list_files(directory_path, max_count, abspath)
  local l = {}
  for fn in lfs.dir(directory_path) do
    if max_count and #l >= max_count then
      break
    end
    local full_fn = path.join(directory_path, fn)
    if lfs.attributes(full_fn, 'mode') == 'file' then
      table.insert(l, abspath and full_fn or fn)
    end
  end
  return l
end

function clamp(x, lo, hi)
  return math.max(math.min(x, hi), lo)
end

function saturate(x)
  return clam(x, 0, 1)
end

function lerp(a, b, t)
  return (1-t) * a + t * b
end

function shuffle_n(array, count)
  count = math.max(count, count or #array)
  local r = #array    -- remaining elements to pick from
  local j, t
  for i=1,count do
    j = math.random(r) + i - 1
    t = array[i]    -- swap elements at i and j
    array[i] = array[j]
    array[j] = t
    r = r - 1
  end
end

function shuffle(array)
  local i, t
  for n=#array,2,-1 do
    i = math.random(n)
    t = array[n]
    array[n] = array[i]
    array[i] = t
  end
  return array
end

function shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

function deep_copy(obj, seen)
  if type(obj) ~= 'table' then
    return obj
  end
  if seen and seen[obj] then
    return seen[obj]
  end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do
    res[deep_copy(k, s)] = deep_copy(v, s)
  end
  return res
end

function reverse(array)
  local n = #array, t
  for i=1,n/2 do
    t = array[i]
    array[i] = array[n-i+1]
    array[n-i+1] = t
  end
  return array
end

function remove_tail(array, num)
  local t = {}
  for i=num,1,-1 do
    t[i] = table.remove(array)
  end
  return t, array
end

function keys(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, k)
  end
  return l
end

function values(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, v)
  end
  return l
end

function draw_rectangle(img, rect, color, label)
  label = label  or ""
  local sz = img:size()

  local x0 = math.max(1, rect.minX)
  local x1 = math.min(sz[3], rect.maxX)
  local w = math.floor(x1) - math.floor(x0)
  if w >= 0 then
    local v = color:view(3,1):expand(3, w + 1)
    if rect.minY > 0 and rect.minY <= sz[2] then
      img[{{}, rect.minY, {x0, x1}}] = v
    end
    if rect.maxY > 0 and rect.maxY <= sz[2] then
      img[{{}, rect.maxY, {x0, x1}}] = v
    end
  end

  local y0 = math.max(1, rect.minY)
  local y1 = math.min(sz[2], rect.maxY)
  local h = math.floor(y1) - math.floor(y0)
  if h >= 0 then
    local v = color:view(3,1):expand(3, h + 1)
    if rect.minX > 0 and rect.minX <= sz[3] then
      img[{{}, {y0, y1}, rect.minX}] = v
    end
    if rect.maxX > 0 and rect.maxX <= sz[3] then
      img[{{}, {y0, y1}, rect.maxX}] = v
    end
  end

  if h > 0 and w > 0 and x0>10 and y0 < y1-10 then
    --print(string.format("drawText '%s' at x0 = %f, y0 = %f,info: x1 = %f, y1 = %f, image width = %f, height = %f ",label,x0,y0,x1,y1,sz[3],sz[2]))
    img:copy(image.drawText(img:double(),label,x0,y0,{color = {color[1]*255,color[2]*255,color[3]*255}, size =1}))
    img:cuda()
  end
  return img
end


function remove_quotes(s)
  return s:gsub('^"(.*)"$', "%1")
end

function normalize_debug(t)
  local lb, ub = t:min(), t:max()
  return (t -lb):div(ub-lb+1e-10)
end

function find_target_size(orig_w, orig_h, target_smaller_side, max_pixel_size)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    w = math.min(orig_w * target_smaller_side/orig_h, max_pixel_size)
    h = math.floor(orig_h * w/orig_w + 0.5)
    w = math.floor(w + 0.5)
  else
    -- width is smaller than height, set w to target_size
    h = math.min(orig_h * target_smaller_side/orig_w, max_pixel_size)
    w = math.floor(orig_w * h/orig_h + 0.5)
    h = math.floor(h + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end

function find_scaled_size(orig_w, orig_h, target_smaller_side)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    h = target_smaller_side
    w = math.floor(orig_w * h/orig_h + 0.5)
  else
    -- width is smaller than height, set w to target_size
    w = target_smaller_side
    h = math.floor(orig_h * w/orig_w + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end

function load_image(fn, color_space, base_path)
  if not path.isabs(fn) and base_path then
    fn = path.join(base_path, fn)
  end
  local img = image.load(fn, 3, 'float')
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end
  return img
end

function printf(...)
  print(string.format(...))
end

function follow_iou(gt_masks, mask, array_classes_gt_objects, last_matrix, available_objects)
    results = torch.zeros(#array_classes_gt_objects,1)--[np.size(array_classes_gt_objects), 1])
    for k=1,#array_classes_gt_objects do-- in range(np.size(array_classes_gt_objects)):
          if available_objects[k] == 1 then
                gt_mask = gt_masks[k]
                iou = Rect.IoU(mask, gt_mask)
                results[k] = iou
          else
                results[k] = -1
          end
    end
    max_result,ind = torch.max(results,1)
    --ind = np.argmax(results)
    print(ind)
    iou = last_matrix[ind]
    
    new_iou = max_result
    return iou, new_iou, results, ind
  end
  
  function get_true_action(gt_mask,region_mask,scale_subregion)
    local x= {}
    sub_width = region_mask:width()*scale_subregion
    sub_height = region_mask:height()*scale_subregion
    x[1]=Rect.fromXYWidthHeight(region_mask.minX, region_mask.minY, sub_width, sub_height) --sub_region1
    x[2] =Rect.fromXYWidthHeight(region_mask.maxX-sub_width,region_mask.minY,sub_width,sub_height)--sub_region2
    x[3]=Rect.fromXYWidthHeight(region_mask.minX,region_mask.maxY-sub_height,sub_width,sub_height)--sub_region3 
    x[4]=Rect.fromXYWidthHeight(region_mask.maxX-sub_width,region_mask.maxY-sub_height,sub_width,sub_height)--sub_region4
    x[5]=Rect.fromXYWidthHeight((region_mask.minX+region_mask.maxX-sub_width)/2,(region_mask.minX+region_mask.maxY-sub_height)/2,sub_width,sub_height)--sub_region5
    x[6] = region_mask--sub_region6 
    local iou = torch.zeros(6)
    for i=1,6 do
      iou[i]=Rect.IoU(gt_mask,x[i])
      print(string.format('iou of action %d is %f',i,iou[i]))
    end
    local max_iou ,max_idx = torch.max(iou,1)
    if iou[6]>0.7 then
      return 6
    else
      return max_idx
    end
  end
  
  

    ------test
--    local fn = '/home/shulang/test/test.jpg'
--    local small_side = 100
--    local max_side = 1000
--    local color='rgb'
--    local img ,dim = load_image_auto_size(fn,small_side,max_side)
--    print('image size ',img:size())
--    local gt_mask = Rect.new(10,10,40,40)
--    local region_mask = Rect.new(12,12,100,100)
--    local scale_subregion = 3.0/4
--    tr_action = get_true_action(gt_mask,region_mask,scale_subregion)
--    print('true action =',tr_action)