-- specify the base path of the ILSVRC2015 dataset: 
--ILSVRC2015_BASE_DIR = '/home/chales/data/ILSVRC2015/ILSVRC2015/'
CSRN_BASE_DIR = '/home/ubuntu/data/'


require 'path'
require 'lfs'
require 'LuaXML'      -- if missing use luarocks install LuaXML
require 'utilities'
require 'Rect' 

local ground_truth = {}
local class_names = {}
local class_index = {}
local type_names = {}
local type_index = {}

function import_file(anno_base, data_base, fn, name_table)
  local x = xml.load(fn)
  local a = x:find('annotation')
  local folder = a:find('folder')[1]
  local filename = a:find('filename')[1]
  local src = a:find('source')
  local db = src:find('database')[1]
  local sz = a:find('size')
  local w = tonumber(sz:find('width')[1])
  local h = tonumber(sz:find('height')[1])
  
  -- generate path relative to annotation dir and join with data dir
  local dirpath = data_base 
  local image_path = path.join(dirpath, path.relpath(fn, anno_base))  
  
  -- replace 'xml' file ending with 'JPEG'
  image_path = string.sub(image_path, 1, #image_path - 3) .. 'jpg'    
  table.insert(name_table, image_path)
  
  for _,e in pairs(a) do
    if e[e.TAG] == 'object' then
    
      local obj = e
      local color = obj:find('color')[1]
      local dir = obj:find('dir')[1]
      local type0 = obj:find('type')
      local type1 = type0:find('type1')[1]
      local type2 = type0:find('type2')[1]
      local type3 = type0:find('type3')[1]
      
      local bb = obj:find('objectbox') 
      local objxmin = tonumber(bb:find('xmin')[1])
      local objxmax = tonumber(bb:find('xmax')[1])
      local objymin = tonumber(bb:find('ymin')[1])
      local objymax = tonumber(bb:find('ymax')[1])
      
      bb = obj:find('platebox') 
      local pxmin = tonumber(bb:find('xmin')[1])
      local pxmax = tonumber(bb:find('xmax')[1])
      local pymin = tonumber(bb:find('ymin')[1])
      local pymax = tonumber(bb:find('ymax')[1])
      
      bb = obj:find('typebox') 
      local txmin = tonumber(bb:find('xmin')[1])
      local txmax = tonumber(bb:find('xmax')[1])
      local tymin = tonumber(bb:find('ymin')[1])
      local tymax = tonumber(bb:find('ymax')[1])
      
      local pcon = obj:find('platecontent')[1]
      
      
      local name = 'car'   --  color .. type1 .. type2 .. dir
      
--      local tmpnum=string.find(fn,'0000003')
--      if tmpnum then
--        print('****' .. fn)    
--      end
      if name == '其他' then 
          name ='其它' 
      end
      if not class_index[name] then
        class_names[#class_names + 1] = name
        class_index[name] = #class_names 
      end 
      if not type_index[type3] then
        type_names[#type_names + 1] = type3
        type_index[type3] = #type_names 
      end 
     
      local roi = {
        rect = Rect.new(objxmin, objymin, objxmax, objymax),
        class_index = class_index[name],
        class_name = name,
        type_index = type_index[type3],
        type_name = type3,
        prect=Rect.new(pxmin, pymin, pxmax, pymax),
        trect=Rect.new(txmin, tymin, txmax, tymax),
        pcon = pcon
      }
      
      local file_entry = ground_truth[image_path]
      if not file_entry then
        file_entry = { image_file_name = image_path, rois = {} }
        ground_truth[image_path] = file_entry
      end 
      table.insert(file_entry.rois, roi)
    end
  end
end

function import_directory(anno_base, data_base, directory_path, recursive, name_table)
   for fn in lfs.dir(directory_path) do
    local full_fn = path.join(directory_path, fn)
    local mode = lfs.attributes(full_fn, 'mode') 
    if recursive and mode == 'directory' and fn ~= '.' and fn ~= '..' then
      import_directory(anno_base, data_base, full_fn, true, name_table)
      collectgarbage()
    elseif mode == 'file' and string.sub(fn, -4):lower() == '.xml' then
      print(full_fn)
      import_file(anno_base, data_base, full_fn, name_table)
    end
    if #ground_truth > 10 then
      return
    end
  end
  return l
end

-- recursively search through training and validation directories and import all xml files
function create_ground_truth_file(dataset_name, base_dir, train_annotation_dir, val_annotation_dir, train_data_dir, val_data_dir, background_dirs, output_fn)
  function expand(p)
    return path.join(base_dir, p)
  end
  
  local training_set = {}
  local validation_set = {}
  import_directory(expand(train_annotation_dir), expand(train_data_dir), expand(train_annotation_dir), true, training_set)
  import_directory(expand(val_annotation_dir), expand(val_data_dir), expand(val_annotation_dir), true, validation_set)
  local file_names = keys(ground_truth)
  
  -- compile list of background images
  local background_files = {}
  for i,directory_path in ipairs(background_dirs) do
    directory_path = expand(directory_path)
    for fn in lfs.dir(directory_path) do
      local full_fn = path.join(directory_path, fn)
      local mode = lfs.attributes(full_fn, 'mode')
      if mode == 'file' and string.sub(fn, -5):lower() == '.jpeg' then
        table.insert(background_files, full_fn)
      end
    end
  end
  
  print(string.format('Total images: %d; classes: %d; types: %d;  train_set: %d; validation_set: %d; (Background: %d)', 
    #file_names, #class_names, #type_names,#training_set, #validation_set, #background_files
  ))
 for i=1,#class_names do
   print(class_names[i])
 end
  save_obj(
    output_fn,
    {
      dataset_name = dataset_name,
      ground_truth = ground_truth,
      training_set = training_set,
      validation_set = validation_set,
      class_names = class_names,
      type_index = type_index,
      type_names = type_names,
      class_index = class_index,
      background_files = background_files
    }
  )
  
  print('Done.')
  
end
path=dofile('path.lua')

background_folders = {}
table.insert(background_folders, 'Background')


create_ground_truth_file(
  'CSRN',
   CSRN_BASE_DIR,
  'Annotations/train', 
  'Annotations/val',
  'Data/train',
  'Data/val',
  background_folders,
  'logs/CSRN_DET.t7'
)
