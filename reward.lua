--require 'features'
require 'torch'
require 'nn'
require 'nngraph'

local number_of_actions =6
local actions_of_history =4
local reward_movement_action =1
local reward_teminal_action =3
local iou_threshold =0.5


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