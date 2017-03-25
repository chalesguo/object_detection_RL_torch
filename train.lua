local torch = require 'torch'
local cutorch = require 'cutorch'
local pl = require 'pl'
local lfs = require 'lfs'
local optim = require 'optim'
local image = require 'image'
local cunn = require 'cunn'
local c = require 'trepl.colorize'
local gnuplot = require 'gnuplot'


require 'nngraph'
require 'utilities'
require 'measure'
require 'fileIo'
require 'reward'
require 'models/vggorg_small'

-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-cfg', 'config/imagenetorg.lua', 'configuration file')
cmd:option('-name', 'imgnet', 'experiment name, snapshot prefix')
cmd:option('-train', 'logs/CSRN_DET.t7', 'training data file name')
cmd:option('-restore', '', 'network snapshot file name to load') --logs/imgnet_009000.t7
cmd:option('-mode', 'both', 'one of three training modes: onlyPnet, onlyCnet, or both')
cmd:option('-snapshot', 200, 'snapshot interval')
cmd:option('-plot', 100, 'plot training progress interval')
cmd:option('-lr', 1E-4, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'sgd', 'Optimizer')
cmd:option('-resultDir', 'logs', 'Folder for storing all result. (training progress etc.)')
cmd:option('-scale_subregion',3.0/4.0,'the scale of subregion')
cmd:option('-scale_mask',1.0/3.0,'th scale of the mask')
cmd:option('-max_steps',10,'how many steps of the angent can run before find one object')
cmd:option('-epochs',500,'the epochs of training ')
cmd:option('batch_size',100,'batch to train')
cmd:option('buffer-experience_replay',1000,'capacity of 100 experiences')

cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')

print('Command line args:')
local opt = cmd:parse(arg or {})
print(opt)

print('Options:')
local cfg = dofile(opt.cfg)
print(cfg)

local layer_index = 5
local reward =0
local training_data_filename = opt.train
local epsilon = cfg.epsilon
print("loading model==>")
local model = model_create(cfg)
print(model.DQN_net)
print(model.feature_net)
graph.dot(model.DQN_net.fg,'DQN_net.fg','DQN_net.fg')
graph.dot(model.feature_net.fg,'feature_net.fg','feature_net.fg')
graph.dot(model.DQN_net.bg,'DQN_net.bg','DQN_net.bg')
graph.dot(model.feature_net.bg,'feature_net.bg','feature_net.bg')
print("load model finished")

print('Reading training data file \'' .. training_data_filename .. '\'.')

local training_data = load_obj(training_data_filename)

local image_names = keys(training_data.ground_truth)

print(string.format("Training data loaded:\n Dataset: '%s'; Total files: %d; Train files:%d,Test files:%d;classes: %d; Background: %d",
    training_data.dataset_name,
    #image_names,
    #training_data.training_set,
    #training_data.validation_set,
    #training_data.class_names,
    #training_data.background_files))

--local labels = training_data.ground_truth.roi
torch.setdefaulttensortype('torch.DoubleTensor')
cutorch.setDevice(opt.gpuid+1)
torch.setnumthreads(opt.threads)
if opt.seed~=0 then
    torch.manualSeed(opt.seed)
    cutorch.manualSeed(opt.seed)
end

optimState = {
    learningRate = opt.lr,
    weightDecay = 1E-5,
    momentum = 0.8,
    nesterov = true,
    learningRateDecay = 0,
    dampening =0.0
}
optimMethod = optim.sgd
learnSchedule = {
    {1,8000,5e-4,5e-5},
    {8001,16000,1e-4,1e-5}
}

--local initial_feature_maps = calculate_all_initial_feature_maps(images,model_vgg,image_names)
local amp = nn.SpatialAdaptiveMaxPooling(kw,kh):cuda()

local train_image_names = training_data.training_set

for i=1,opt.epochs do
  for j=1,#train_image_names do
    print(string.format('epoch :%d',i))
    image_name = train_image_names[j]
    img = image.load(image_name)
    --img = load_image_auto_size(image_name,cfg.target_smaller_side,cfg.max_pixel_size,cfg.color_space)
  
    local annotation ={}
    annotation = training_data.ground_truth[image_name]
    print(annotation)
    print('num of annotation.rois',#annotation.rois)
    arrary_classes_gt_objects=annotation.rois

    for k=1,#arrary_classes_gt_objects do
        gt_mask=annotation.rois[k].rect
        region_mask=Rect.new(0,0,img:size(2),img:size(3))        
        shape_gt_masks=gt_mask:size()
        if k>1 then
                tmp_mask=annotation.rois[k-1].rect
                img.sub(tmp_mask[1],tmp_mask[2],tmp_mask[3],tmp_mask[4])[1]=0.485--将检测到的目标用均值填充，该
                img.sub(tmp_mask[1],tmp_mask[2],tmp_mask[3],tmp_mask[4])[2]=0.456--均值数据来自于pytorch论坛
                img.sub(tmp_mask[1],tmp_mask[2],tmp_mask[3],tmp_mask[4])[3]=0.406 
        end
        feature_maps =model.feature_net:forward(img)
        local aga=true
        history_vector= torch.zeros(24)
        history_vector[6]=1
        history_vector[12]=1
        history_vector[18]=1
        history_vector[24]=1
        scale_mask=3/4
        region_coordinates = region_mask
        while aga do
          step=step+1
            if (i<100) and (new_iou>0.5) then
                action =6
            elseif math.random()<epsilon then
                action = torch.random(1,6)
            else 
                action,_ = torch.max(qval) 
            end
          region_descriptor = feature_maps[1].crop(inputToFeatureRect(region_coordinates,layer_index))
          region_descriptor_2=amp:forward(region_descriptor):view(512*kw*kh)
          state = torch.cat(history_vector,region_descriptor_2)
          qval = model.DQN_net:forward(state)
          action, m_index = torch.max(qval) 
--          true_index=get_true_action(gt_mask,region_mask)
--          if m_index==true_index then
--            tqval[m_index]=qval[m_index]+0.1
--          else
--            tqval[m_index]=qval[m_index]-0.1
--          end
          if action==6 then
              aga=false
          else
                sub_width = region_coordinates:width()*scale_mask
                sub_height= region_coordinates:height()*scale_mask
                if action ==1 then
                    region_coordinates=Rect.fromXYWidthHeight(region_coordinates.minX, region_coordinates.minY, sub_width, sub_height)
                elseif action ==2 then
                    region_coordinates=Rect.fromXYWidthHeight(region_coordinates.maxX-sub_width,region_coordinates.minY,sub_width,sub_height)
                elseif action ==3 then
                    region_coordinates=Rect.fromXYWidthHeight(region_coordinates.minX,region_coordinates.maxY-sub_height,sub_width,sub_height)
                elseif action ==4 then
                    region_coordinates=Rect.fromXYWidthHeight(region_coordinates.maxX-sub_width,region_coordinates.maxY-sub_height,sub_width,sub_height)
                elseif action ==5 then
                    region_coordinates=Rect.fromXYWidthHeight((region_coordinates.minX+region_coordinates.maxX-sub_width)/2,(region_coordinates.minX+region_coordinates.maxY-sub_height)/2,sub_width,sub_height)
                end
          end
--          local f = criterion:forward(qval,  tqval)
--          local df_do = criterion:backward(qval,  tqval)
--          model.DQN_net:backward(inputs, df_do)
           iou,new_iou,last_matrix,index = follow_iou(gt_masks,region_mask,arrary_classes_gt_objects,class_object,last_matrix,available_objects)
          gt_mask=gt_masks[index]
          reward =get_reward_movement(iou,new_iou)
          iou = new_iou
          local action_vect=torch.zero(cfg.action_count)
          action_vect[action]=1
          history_vector = torch.cat(history_vector:narrow(1,#history_vector-6),action)--更新历史行动
          if masked==1 then
            for p=1,#gt_masks do
                overlap = Rect.IoU(old_region_mask,gt_masks[p])
                if overlap>0.6 then
                    available_objects[p]=0
                end
            end
          end
        if available_objects:sum()==0 then
            not_finished=0
        end
        gt_mask=gt_masks[index]
        status =1
        action =0
        if step < opt.max_steps then
            step=step+1
        end
            if #replay <buffer_experience_replay then
                table.insert(replay,{state,action,reward,new_state})
            else
                if h<buffer_experience_replay-1 then
                    h=h+1
                else
                    h=0
                end
                h_aux=h:int()
                replay[h_aux]={state,action,reward,new_state}
                minibatch = torch.randperm(#replay):spilt(batch_size)[1] --random.sample(replay,batch_size)--todo
                X_train ={}
                y_train={}
                for mem=1,#minibatch do
                    old_state,action.reward,new_state =minibatch[mem]
                    old_qval = model.DQN_net:forward(old_state)
                    newQ = model.DQN_net:forward(new_state)
                    _,maxQ=torch.max(newQ,1)
                    y=torch.zeros(6)
                    y=old_qval
                    if action ~=6 then 
                        update=reward+(gamma*maxQ)
                    else
                        update=reward
                    end
                    y[action]=update
                    table.insert(X_train,old_state)
                    table.insert(y_train,y)
                end
                X_train=X_train[1]
                y_train=y_train[1]
                train(model.DQN_net,X_train,y_train,batch_size,nb_epoch,optimState)-- trian code
                state = new_state
            end
            if action ==6 then
                status =0
                masked = 1
            else
                masked=0
            end
        end
        available_objects[index]=0
    end
    if epsilon>0.1 then
        epsilon=epsilon -0.1
    end
end
torch.save(opt.save,model)
end