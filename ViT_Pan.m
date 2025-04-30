clear all
close all

%Images Section

%in the labels, 
%for pan_data, 1 is infected, 0 is noninfected
%for nih data, 1 is parasitized, 0 is noninfected
imds_Pan = imageDatastore("dw_Pan_data", 'IncludeSubFolders', true,'FileExtensions',".png", 'LabelSource', 'foldername');

%division of dataset (for now is split according to the "best" division
%used in the paper
[imds_Pan_Train, imds_Pan_Val, imds_Pan_Test] = splitEachLabel(imds_Pan, 0.9, 0.05, 0.05);

%The random rotation and scaling
%(ONLY FOR TRAINING)
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXReflection', true, ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

%implementing messing with the training data
augImds_Pan_Train = augmentedImageDatastore([384 384], imds_Pan_Train, 'DataAugmentation', augmenter);

%resizing the other two dataset subsections
augImds_Pan_Val = augmentedImageDatastore([384 384], imds_Pan_Val);

augImds_Pan_Test = augmentedImageDatastore([384 384], imds_Pan_Test);


%Visual Transformer section

%loading the pretrained model
[net,classNames] = visionTransformer("base-16-imagenet-384");


%selectively freezing all the layers except for the attention layers
layers = layerGraph(net);
layerschanged = customfreezeWeights1(layers);


%adjusting model to produce predictions that were particular to the classes
%in dataset by changing classification head
number_classes = 2; % for 1(infected cells) and 0(non-infected cells)
newLayers = [
    fullyConnectedLayer(number_classes, 'Name', 'new_head')
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')
];

%replacing the final layers in the model’s layer graph.
layerschanged = replaceLayer(layerschanged, 'head', newLayers(1));
% layerschanged = replaceLayer(layerschanged, 'softmax', newLayers(2));

% % Adding the new classification layer
% layerschanged = addLayers(layerschanged, newLayers(3));
% layerschanged = connectLayers(layerschanged, 'new_softmax', 'new_classoutput');

dlnet = dlnetwork(layerschanged);

%Training the Neural network

%First, making optimizer
Pan_options = trainingOptions('adam', ...
    'MiniBatchSize', 6, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 5e-4, ...
    'ValidationData', augImds_Pan_Val, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment','gpu' ...
    ,'ValidationData',augImds_Pan_Val, ...
    ValidationFrequency=35, ...
    ValidationPatience = 1);

trained_Pan_Net = trainnet(augImds_Pan_Train, dlnet, "crossentropy",Pan_options);