require 'features'
require 'torch'
require 'nn'
require 'nngraph'

local number_of_actions =6
local actions_of_history =4
local visual_descriptor_size = 25088
local reward_movement_action =1
local reward_teminal_action =3
local iou_threshold =0.5

function  update_history_vector(history,action)
    action_vector = torch.zeros(number_of_actions)
    action_vector[action]=1
    size_history_vector = history_vector:nonzero():size(1)
    updated_history_vector = torch.Tensor(number_of_actions*actions_of_history)
    if size_history_vector <actions_of_history then
        aux2 =0
        for l =number_of_actions*size_history_vector,number_of_actions*size_history_vector+number_of_actions-1 do
            history_vector[l] = action_vector[aux2]
            aux2 =aux2 +1
        end
        return history_vector
    else
        for j=1,number_of_actions*(actions_of_history-1) do
            update_history_vector[j] = history_vector[j+number_of_actions]
            aux=0
        end
        for k=number_of_actions*(actions_of_history-1),number_of_actions*actions_of_history do
            update_history_vector[k] = action_vector[aux]
            aux = aux+1
        end
        return update_history_vector
    end
end

function get_state(image,history_vector,model_vgg)
    descriptor_image = get_conv_image_descriptor_for_image(image,model_vgg)
    descriptor_image = descriptor_image:resize(visual_descriptor_size,1) --todo
    history_vector = history_vector:resize(number_of_actions*actions_of_history,1)
    state = torch.cat(descriptor_image,history_vector)
end

function get_reward_movement(iou,new_iou)
    if new_iou >iou then
        reward = reward_movement_action
    else
        reward = - reward_movement_action
    end
    return reward
end

function get_reward_trigger(new_iou)
    if new_iou >iou_threshold then
        reward = reward_teminal_action
    else
        reward = -reward_teminal_action
    end
    return reward
end

function get_q_network(weights_path)
    local net = nn.Sequential()
    net :add(nn.Dropout(0.2))
    net :add(nn.Linear(25122,1024))
    net :add(nn.BatchNormalization(1024))
    net : add(nn.ReLU(true))
    net :add(nn.Dropout(0.2))
    net :add(nn.Linear(1024,1024))
    net :add(nn.BatchNormalization(1024))
    net :add(nn.ReLU(true))
    net :add(Dropout(0.2))
    net :add(nn.Linear(1024,6))
end

function get_array_of_1_networks_for_pascal(weights_path,class_object):
    local model = get_q_network(weights_path)
    return model
end

        