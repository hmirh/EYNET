%% ------------------------------------------------------------------------
clc;clear all;close all;
addpath(genpath('./'));
%% ------------------------------------------------------------------------
global flag % Type of augmentation
train = 1; % 1 train network, 0 use pretrained network
newData = 0;
estimateanchor = 0; % Estimate Anchor Boxes From Training Data (The use of anchor boxes enables a network to detect multiple objects, objects of different scales, and overlapping objects)
anchorid = 6;
networkInputSize = [224 224 3];
classNames = {'airport'};
DetectionNetworkSource = {'Conv_1_bn','Conv_1_bn','res5b','res5b'}; %Feature extraction layers from earlier in the network have higher spatial resolution but may extract less semantic information compared to layers further down the network. Choosing the optimal detection network source requires trial and error
load('Models\InitialModel.mat');
net = InitialModel;
net = ShearFilters(net);
baseNetwork = dlnetwork(net); 
numEpochs = 30;
validationfrequency = 100;
validationpatience = 100;
miniBatchSize = 8;
learningRate = 0.001; % A low learning rate will cause your model to converge very slowly. A high learning rate will quickly decrease the loss in the beginning but might have a hard time finding a good solution.
l2Regularization = 0.00001; %The most common values of the regularization parameter are often on a logarithmic scale between 0 and 0.1, such as 0.1, 0.001, 0.00001 etc.
penaltyThreshold = 0.5; %Specify the penalty threshold as 0.5. Detections that overlap less than 0.5 with the ground truth are penalized.
averageSqGrad = [];
maxrange = learningRate;
minrange = 0.0001;
iterperepoch = 100;
numObservations = iterperepoch * numEpochs;
%% ------------------------------------------------------------------------
stopcounter = 0;
ap = 0;
recall = 0;
precision = 0;
ACC = 0;
%% ------------------------------------------------------------------------
if newData == 1
    load('Data\Data.mat');
    Data(1328:end, :) = [];
    rng(0);
    shuffledIndices = randperm(height(Data));
    idx1 = floor(0.6 * length(shuffledIndices));% 60% Train
    idx2 = floor(0.7 * length(shuffledIndices));% 10% Validation 30% Test
    trainingDataTbl = Data(shuffledIndices(1:idx1), :);
    validationDataTbl = Data(shuffledIndices(idx1+1:idx2), :);
    testDataTbl = Data(shuffledIndices(idx2+1:end), :);
    % ----------------
    bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
    bldsValidation = boxLabelDatastore(validationDataTbl(:, 2:end));
    bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));
    % ----------------
    imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
    imdsValidation = imageDatastore(validationDataTbl.imageFilename);
    imdsTest = imageDatastore(testDataTbl.imageFilename);
    % ----------------
    trainingData = combine(imdsTrain, bldsTrain);
    validationData = combine(imdsValidation, bldsValidation);
    testData = combine(imdsTest, bldsTest);
    % ----------------
    % validateInputData(trainingData);
    % validateInputData(validationData);
    % validateInputData(testData);
    % ----------------
    save('Data\trainingData.mat','trainingData');
    save('Data\validationData.mat','validationData');
    save('Data\testData.mat','testData');
else
    load('Data\trainingData.mat');
    load('Data\validationData.mat');
    load('Data\testData.mat');
end
%% ------------------------------------------------------------------------
if estimateanchor
    rng(0);
    trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
    maxNumAnchors = 100;
    meanIoU = zeros([maxNumAnchors,1]);
    anchorBoxes = cell(maxNumAnchors, 1);
    parfor k = 1:maxNumAnchors
        [anchorBoxes{k}, meanIoU(k)] = estimateAnchorBoxes(trainingDataForEstimation, k);
    end
    save('AnchorBoxes.mat','anchorBoxes');
    save('meanIoU.mat','meanIoU');
    figure;
    plot(1:maxNumAnchors,meanIoU,'-o');
    ylabel("Mean IoU");
    xlabel("Number of Anchors");
    title("Number of Anchors vs. Mean IoU");
    anchors = anchorBoxes{anchorid};
else
    load('AnchorBoxes.mat');
    anchors = anchorBoxes{anchorid};
end
% ---------------- Use larger anchors at lower scale and smaller anchors at higher scale
% area = anchors(:, 1).*anchors(:, 2);
% [~, idx] = sort(area, 'descend');
% anchors = anchors(idx, :);
% w = floor(length(anchors)/length(DetectionNetworkSource));
% c = 1;
% for i = 1 : length(DetectionNetworkSource)
%     if i == length(DetectionNetworkSource)
%         DivanchorBoxes{i,1} = anchors(c:end,:);
%         break;
%     end
%     DivanchorBoxes{i,1} = anchors(c:i*w,:);
%     c = c + w;
% end
DivanchorBoxes{1,1} = anchors;
DivanchorBoxes{2,1} = anchors;
DivanchorBoxes{3,1} = anchors;
DivanchorBoxes{4,1} = anchors;
%% ------------------------------------------------------------------------
augmentedTrainingData = transform(trainingData, @augmentData);
augmentedTrainingData = transform(augmentedTrainingData, @(data)preprocessData(data, networkInputSize));
validationData = transform(validationData, @(data)preprocessData(data, networkInputSize));
testData = transform(testData, @(data)preprocessData(data, networkInputSize));
%% ------------------------------------------------------------------------
if train == 0
    load('Models\EYNetModel.mat')
else
    yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, ...
        DivanchorBoxes, 'DetectionNetworkSource', DetectionNetworkSource, 'ModelName','EYNet');
    analyzeNetwork(yolov3Detector.Network);

    mbqTrain = minibatchqueue(augmentedTrainingData, 2, ...
        "MiniBatchSize", miniBatchSize,...
        "OutputEnvironment", "gpu",...
        "DispatchInBackground", false ,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "OutputCast", ["", "double"]);

    mbqValidation = minibatchqueue(validationData, 2, ...
        "MiniBatchSize", miniBatchSize,...
        "OutputEnvironment", "gpu",...
        "DispatchInBackground", false ,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "OutputCast", ["", "double"]);

    fig = figure;
    [plotap,plotrecall,plotprecision, plotdetectrate, lossPlotter,lossPlotter2, VallossPlotter, learningRatePlotter] = ...
        configureTrainingProgressPlotter(fig);
    iteration = 0;
    lossInfoV.totalLoss = 0;
    preLoss = -inf;
    apar = -inf; recallar = 0;precisionar = 0;F1scorear = 0;ACCar = 0;logAverageMissRatear=0;
    val = 1;    
    error = zeros(1, iterperepoch);    
    flag = 0;    
    for epoch = 1:numEpochs
        index = 0;
        reset(mbqTrain);shuffle(mbqTrain);
        while(index < iterperepoch)
            %% ------------------------------------------------------------
            index = index + 1;
            iteration = iteration + 1;
            flag = 0.7 - LRWarmRestarts(iteration, numEpochs, 0, 1, numEpochs, numObservations, 1);
            submbq = partition(mbqTrain, iterperepoch, index);
            commandwindow;
            %% ------------------------------------------------------------
            currentLR = LRWarmRestarts(iteration, numEpochs, minrange, maxrange, 3, numObservations, 10);    %Cosine Annealing Learning Rate Schedule
            reset(submbq);shuffle(submbq);
            [XTrain, YTrain] = next(submbq);
            % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
            [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
            % Apply L2 regularization.
            gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
            % Update the detector learnable parameters using the rmsprop optimizer.
            [yolov3Detector.Learnables, averageSqGrad] = rmspropupdate(yolov3Detector.Learnables, gradients, averageSqGrad, currentLR); % yolov3Detector.Network
            yolov3Detector.State = state;
            error(1, index) = double(gather(extractdata(lossInfo.totalLoss)));
            lossMean = updatePlots(ap, recall, precision, ACC, plotap, plotrecall, plotprecision, plotdetectrate, lossPlotter, lossPlotter2, VallossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss, lossInfoV.totalLoss);
            displayLossInfo(epoch, iteration, currentLR, lossInfoV.totalLoss, lossInfo, lossMean);

            if index > 5
                errorAvg = movmean(error, 5);
                errorStd = movstd(error, 5);
                i = index - 5;
                if (errorAvg(1, i) ) < error(1 , index)
                    flag = 2;
                    reset(submbq);shuffle(submbq);
                    [XTrain, YTrain] = next(submbq);
                    [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
                    gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
                    [yolov3Detector.Learnables, averageSqGrad] = rmspropupdate(yolov3Detector.Learnables, gradients, averageSqGrad, currentLR);
                    yolov3Detector.State = state;
                end
            end
            %% ----------------------------------------------------------------
            if mod(iteration, validationfrequency) == 0
                reset(mbqValidation);shuffle(mbqValidation);
                sumloss = 0;
                count = 0;
                while(hasdata(mbqValidation))
                    [XVal, YVal] = next(mbqValidation);
                    [gradientsV, stateV, lossInfoV] = dlfeval(@modelGradients, yolov3Detector, XVal, YVal, penaltyThreshold);
                    sumloss = sumloss + lossInfoV.totalLoss;
                    count = count + 1;
                end
                lossInfoV.totalLoss = sumloss / count;
                lossInfoV.totalLoss = double(extractdata(gather(lossInfoV.totalLoss)));
                %% --------------------------------------------------------
                results = detect(yolov3Detector, validationData, 'DetectionPreprocessing', 'none', 'MiniBatchSize',  miniBatchSize);                
                res = EvaluateModelMain(results,validationData, networkInputSize, [], '', 1);
                ap = res.ap;
                recall = res.recall;
                precision = res.precision; 
                F1score = res.F1score;
                ACC = res.ACC;
                logAverageMissRate = res.logAverageMissRate;
                %% --------------------------------------------------------
                if ap < max(apar) %lossInfoV.totalLoss
                    stopcounter = stopcounter + 1;
                    if stopcounter == validationpatience
                        save(strcat('D:\EYNetModel\',num2str(epoch), 'yolov3Detector.mat'),'yolov3Detector');
                        disp('training stop: the validation loss stops decreasing');
                        return;
                    end
                else
                    stopcounter = 0;
                end
                save(strcat('D:\EYNetModel\',num2str(epoch), 'yolov3Detector.mat'),'yolov3Detector');
                apar(1, val) = res.ap;               
                recallar(1, val) = res.recall;
                precisionar(1, val) = res.precision; 
                F1scorear(1, val) = res.F1score;
                ACCar(1, val) = res.ACC;
                logAverageMissRatear(1, val) = res.logAverageMissRate;
                preLoss(1, val) = lossInfoV.totalLoss;
                val = val + 1;
            end
        end
    end
end
%% ------------------------------------------------------------------------
results = detect(EYNetModel, testData, 'DetectionPreprocessing', 'none', 'MiniBatchSize',  miniBatchSize);
groundTruthBboxes = testData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 2}.LabelData(:,1);
EvaluateModelMain(results,testData, networkInputSize, groundTruthBboxes, 'EYNet', 0);
%% ------------------------------------------------------------------------
data = read(testData);
I = data{1, 1};
[bboxes, scores, labels] = detect(EYNetModel, I, 'DetectionPreprocessing', 'none');
overlapRatio = bboxOverlapRatio(bboxes, data{1, 2});
text = strcat('S:', num2str(round(scores*100, 2)), ', OR:', num2str(round(overlapRatio*100, 2)));
I = insertShape(mat2gray(I),'FilledRectangle',data{1, 2},'Color','white','LineWidth', 2,'Opacity',0.5);
I = insertObjectAnnotation(I,'rectangle',bboxes,text, 'color', 'white', 'LineWidth', 2, 'FontSize', 14, 'Font', 'Arial Bold');
figure;imshow(I, []);
%% ------------------------------------------------------------------------