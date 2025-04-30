% Get test images and labels
testImgs = augImds_Pan_Test;
trueLabels = imds_Pan_Test.Labels;
classNames = categories(trueLabels);      % e.g., ["noninfected", "infected"]

% Initialize prediction arrays
YPred = {};
YProbs = {};

% Reset the datastore
reset(testImgs);

while hasdata(testImgs)
    data = read(testImgs);
    images = data.input;  % This is a cell array of images

    for i = 1:numel(images)
        image = images{i};

        if size(image, 3) == 1
            image = repmat(image, 1, 1, 3);
        end

        image = im2single(image);
        dlImg = dlarray(image, 'SSCB');
        if canUseGPU
            dlImg = gpuArray(dlImg);
        end

        dlY = predict(trained_Pan_Net, dlImg);
        probs = extractdata(softmax(dlY));
        [~, idx] = max(probs);

        YPred{end+1,1} = classNames{idx};
    end
end

% Convert to categorical (same type as true labels)
YPred = categorical(YPred, categories(trueLabels));

% === Accuracy ===
accuracy = mean(YPred == trueLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% === Confusion Matrix ===
figure;
confusionchart(trueLabels, YPred);
title('Confusion Matrix for Test Set');
