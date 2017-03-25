require 'torch'
require 'nn'
local feature_size = 7
scale_reduction_shallower_feature = 16
scale_reduction_deeper_feature =32
factor_x_input = 1.0
factor_y_input = 1.0

function get_feature_maps(model,img)
    return {get_feature_map_4(model,img),get_feature_map_8(model,img)}
end

function get_feature_map_8(model,img)
    img[1] = img[1] - img[1]:mean()
    img[2] = img[2] - img[2]:mean()
    img[3] = img[3] - img[3]:mean()
    img = np.expand_dims(im,axis =1) --todo
    inputs = {K.learning_phase()} + model.inputs
    _convout1_f = K.function(inputs,model.layers[23].output)
    feature_map = _convout1_f(0 + img)
    feature_map = feature_map[1,1,1]
    return feature_map
end

function crop_roi(feature_map,coordinates)
    return feature_map:select(2,{coordinates[1],coordinates[3]}
end

function obtain_descriptor_from_feature_map(feature_map,region_coordinates)
    initial_width = region_coordinates[3]*factor_x_input
    initial_height = region_coordinates[4]*factor_x_input
    scale_aux = math.sqrt(initial_height*initial_width)/math.sqrt(feature_size*feature_size)
    if scale_aux >scale_reduction_deeper_feature then
        scale = scale_reduction_deeper_feature
        feature_map = feature_maps[2]
    else
        scale = scale_reduction_shallower_feature
        feature_map = feature_maps[1]
    end
    new_width = initial_width/scale
    new_height = initial_height/scale
    if new_width <feature_size then
        new_width = feature_size
    end
    if new_height <feature_size then
        new_height = feature_size
    end
    xo = region_coordinates[1]/scale
    yo = region_coordinates[2]/scale
    feat = feature_map
    if new_width +xo > feat:size(3) then
        xo = feat:size(3) -new_width
    end
    if new_height +yo >feat:size(4) then
        y0 = feat:size(4) - new_height
    end
    if xo <0 then
        xo = 0
    end
    if yo <0  then
        yo =0
    end
    new_coordinates = torch.Tensor(xo,yo,new_width,new_height)
    roi = crop_roi(feature_map,new_coordinates)
    if roi:size(2) <feature_size and roi:size(3) <feature_size then
        features = interpolate_3d_features(roi)
    elseif roi:size(3)<feature_size then
        features = interpolate_3d_features(roi)
    elseif roi.shape[2]<feature_size then
        feature = interpolate_3d_features(roi)
    else 
        features = extract_features_from_roi(roi)
    end
    return features
end

function extract_features_from_roi(roi)
    roi_width = roi:size(2)
    roi_height = roi:size(3)
    new_width = roi_width /feature_size
    new_height = roi_height /feature_size
    pooled_values = torch.zeros(feature_size,feature_size,512)
    for j =1,512 do
        for i=1,feature_size do
            for k=1,feature_size do
                if k ==feature_size and i==feature_size then
                    patch = roi[{{j},{i*new_width,roi_width},{k*new_height,roi_height}}]
                elseif k == feature_size then
                    patch = roi[{{j},{i*new_width,(i+1)*new_width},{k*new_height,roi_height}}]
                elseif i ==feature_size then
                    patch = roi[{{j},{i*new_width,roi_width},{k*new_height,(k+1)*new_height}}]
                else
                    patch = roi[{{j},{i*new_width,(i+1)*new_width},{k*new_height,(k+1)*new_height}}]
                end
                pooled_values[i][k][j] = patch:max()
            end
        end
    end
    return pooled_values
end

function get_image_descriptor_for_image(image,model)
    img = image.scale(image,224,224)
    inputs = K.learning_phase() + model.inputs
    _convout1_f = K.function(inputs,model.layers[33].output) --todo
    return _convout1_f(0 + img)
end
