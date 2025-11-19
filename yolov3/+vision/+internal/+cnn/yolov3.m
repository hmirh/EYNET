function detector = yolov3(params)
% YOLOv3 returns YOLO v3 object detector created using the pretrained 
% pretrained YOLO v3 network specified by detectorName. detectorName must
% be 'darknet53-coco' or 'tiny-yolov3-coco'.

% Copyright 2021 The MathWorks, Inc.

if strcmp(params.DetectorName, 'darknet53-coco')
    % Use tripwire to load darknet53-coco network.
    detector = iTripwireMSCOCOModel(params);
else
    % Use tripwire to load tiny-yolov3-coco network.
    detector = iTripwireTinyYOlOv3Model(params);
end

end

% Function to load Darknet53-COCO.
function detector = iTripwireMSCOCOModel(params)

if(isfield(params,'ClassNames'))
    data = load('yolov3COCO.mat');
    network = data.detector.Network;
    anchorBoxes = params.AnchorBoxes;
    if (size(anchorBoxes,1) ~= 3)
        error(message('yolov3:yolov3Detector:numPretrainedAnchorsMismatch',params.ModelName,'3'));
    end
    classes = params.ClassNames;
    layersToReplace = {'conv2d_59','conv2d_67','conv2d_75'};
    network = iconfigureNetwork(network,anchorBoxes,classes,layersToReplace);
    detector = yolov3ObjectDetector(network,classes,anchorBoxes,'ModelName',params.ModelName,'InputSize',params.InputSize);
else
    % Load pretrained network.
    data = load('yolov3COCO.mat');
    detector = data.detector;
end


end

% Function to load Tiny-YOLOv3-COCO.
function detector = iTripwireTinyYOlOv3Model(params)

if(isfield(params,'ClassNames'))
    data = load('tinyYOLOv3COCO.mat');
    network = data.detector.Network;
    anchorBoxes = params.AnchorBoxes;
    if size(anchorBoxes,1) ~= 2
        error(message('yolov3:yolov3Detector:numPretrainedAnchorsMismatch',params.ModelName,'2'));
    end
    classes = params.ClassNames;
    layersToReplace = {'conv2d_10','conv2d_13'};
    network = iconfigureNetwork(network,anchorBoxes,classes,layersToReplace);
    detector = yolov3ObjectDetector(network,classes,anchorBoxes,'ModelName',params.ModelName,'InputSize',params.InputSize);
else
    % Load pretrained network.
    data = load('tinyYOLOv3COCO.mat');
    detector = data.detector;
end


end

function network = iconfigureNetwork(network,anchorBoxes,classes, layersToReplace)
for i = 1:size(layersToReplace,2)
    numAnchors = size(anchorBoxes{1,1},1);
    numClasses = numel(classes);
    numFilters = (numClasses + 5)*numAnchors;
    convOut = convolution2dLayer([1,1],numFilters,'Padding','same','Name',['convOut',num2str(i)]);
    lgraph = layerGraph(network);
    lgraph = replaceLayer(lgraph,layersToReplace{1,i},convOut);
    network = dlnetwork(lgraph);
end
end


function classes = igetYOLOv3Classes()
classes = {'person','bicycle','car','motorbike','aeroplane','bus',...
    'train','truck','boat','traffic light','fire hydrant',...
    'stop sign','parking meter','bench','bird','cat','dog',...
    'horse','sheep','cow','elephant','bear','zebra','giraffe',...
    'backpack','umbrella','handbag','tie','suitcase','frisbee',...
    'skis','snowboard','sports ball','kite','baseball bat',...
    'baseball glove','skateboard','surfboard','tennis racket',...
    'bottle','wine glass','cup','fork','knife','spoon','bowl',...
    'banana','apple','sandwich','orange','broccoli','carrot',...
    'hot dog','pizza','donut','cake','chair','sofa',...
    'pottedplant','bed','diningtable','toilet','tvmonitor',...
    'laptop','mouse','remote','keyboard','cell phone',...
    'microwave','oven','toaster','sink','refrigerator',...
    'book','clock','vase','scissors','teddy bear',...
    'hair drier','toothbrush'};
end
