function lgraph = customfreezeWeights1(lgraph)

    layers = lgraph.Layers;

    for i = 1:numel(layers)
        layer = layers(i);

        % Check if layer even has learnable parameters
        if isprop(layer, "LearnableParameters") && ~isempty(layer.LearnableParameters)

            % Check if it's an attention layer
            isAttention = contains(class(layer), "attention", 'IgnoreCase', true) || ...
                          contains(layer.Name, "mha", 'IgnoreCase', true);

            if ~isAttention
                % Freeze all learnable parameters
                params = layer.LearnableParameters;
                for j = 1:numel(params)
                    paramName = params(j).Name;
                    layer.(paramName + "LearnRateFactor") = 0;
                end
            end
        end

        
        lgraph = replaceLayer(lgraph, layer.Name, layer);
    end

end