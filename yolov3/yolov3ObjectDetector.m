%yolov3ObjectDetector Create YOLO v3 deep learning object detector.
%
% detector = yolov3ObjectDetector(detectorName) loads a pretrained YOLO v3
% object detector specified by detectorName. detectorName must be
% 'darknet53-coco' or 'tiny-yolov3-coco'.
%
% detector = yolov3ObjectDetector(detectorName, classNames, anchorBoxes)
% configures a pretrained YOLO v3 object detector for transfer learning
% with a new set of object classes and anchor boxes. You must train the
% detector for optimal performance.
%
% detector = yolov3ObjectDetector(network, classNames, anchorBoxes)
% creates a YOLO v3 object detector using the input YOLO v3 dlnetwork. Use
% this syntax to specify a custom YOLO v3 network for object detection.
% You must train the detector for optimal performance.
%
% detector = yolov3ObjectDetector(network, classNames, anchorBoxes, 'DetectionNetworkSource', layerName)
% creates a YOLO v3 object detector using the input dlnetwork or DAGNetwork
% object. Use this syntax to automatically add detection subnetworks to the
% input network at layers specified by DetectionNetworkSource. You must
% train the detector for optimal performance.
%
% Inputs:
% -------
%    detectorName   This must be a string that specifies the name of a
%                   pretrained YOLO v3 detector. The name of pretrained
%                   YOLO v3 detector must be one of the following:
%
%                       'darknet53-coco'    A YOLO v3 object detector
%                                           trained on COCO dataset with a
%                                           Darknet-53 base network.
%
%                       'tiny-yolov3-coco'  A YOLO v3 object detector
%                                           trained on COCO dataset using a
%                                           smaller base network.
%
%    network        Specify a dlnetwork or DAGNetwork object. By default,
%                   the network output is used to predict object detections.
%                   The network must have the same number of outputs as the
%                   number of elements in anchorBoxes. To select which
%                   network outputs to use for prediction, use the
%                   DetectionNetworkSource parameter.
%
%    classNames     Specify the names of object classes that the YOLO v3
%                   object detector is configured to detect. classNames can
%                   be a string vector, a categorical vector, or a cell
%                   array of character vectors.
%
%    anchorBoxes    Specify the size of anchor bases as an N-by-1 cell array
%                   of M-by-2 matrices, where each row defines the
%                   [height width] of an anchor box. N is the number of
%                   outputs in the network or the number of names specified
%                   using the DetectionNetworkSource argument.
%
% % Additional input arguments
% ----------------------------
% [...] = yolov3ObjectDetector(..., Name, Value) specifies additional
% name-value pair arguments described below:
%
%   'ModelName'                Detector name specified as string or
%                              character vector.
%
%                              Default: '' or specified detectorName
%
%   'InputSize'                Specify the image sizes to use for detection.
%
%                              Default: network input size
%
%   'DetectionNetworkSource'   Specify an N-by-1 string vector or cell
%                              string, where N are the names of the layers
%                              in input network. The object detection
%                              sub-network is added to these layers. The 
%                              feature extracting layers must be specified
%                              in the reverse of the order in which they
%                              appear in the network architecture. If 
%                              detection network source is not defined
%                              detector assumes the network has a detection
%                              sub-network. If not specified or empty, the
%                              sub-networks are not added to the pretrained
%                              network.
%
%                              Default: {}
%
% yolov3ObjectDetector methods:
%   detect           - Detect objects in an image.
%   preprocess       - Preprocess input data.
%   predict          - Compute detector network output for inference.
%   forward          - Compute detector network output for training.
%
% yolov3ObjectDetector properties:
%   ModelName          - Name of the trained object detector.
%   Network            - YOLO v3 object detection network. (read-only)
%   ClassNames         - A cell array of object class names. (read-only)
%   AnchorBoxes        - Array of anchor boxes. (read-only)
%   InputSize          - Array of image sizes used during training.
%                         (read-only)
%
% Example 1: Detect objects using pretrained YOLO v3.
% ---------------------------------------------------
% % Load pre-trained detector.
% detector = yolov3ObjectDetector('tiny-yolov3-coco');
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labels] = detect(detector, I);
%
% % Display results.
% detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, labels);
% figure
% imshow(detectedImg)
%
% Example 2: Detect objects using custom preprocessing on pretrained YOLO v3.
% ---------------------------------------------------------------------------
% % Load pre-trained detector.
% detector = yolov3ObjectDetector('tiny-yolov3-coco');
%
% % Read test image.
% I = imread('highway.png');
%
% % Resize the test image to network input size.
% I = imresize(I, detector.InputSize);
% I = im2single(I);
%
% % Run detector.
% [bboxes, scores, labels] = detect(detector, I, 'DetectionPreprocessing', 'none');
%
% % Display results.
% detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, labels);
% figure
% imshow(detectedImg)
%
% Example 3: Train a YOLO v3 object detector.
% -------------------------------------------
% %<a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'ObjectDetectionUsingYOLOV3DeepLearningExample')">Object Detection Using YOLO v3 Deep Learning.</a>
%
% See also yolov2ObjectDetector, trainYOLOv2ObjectDetector, fasterRCNNObjectDetector.

% Copyright 2021 The MathWorks, Inc.

classdef yolov3ObjectDetector < vision.internal.cnn.ObjectDetector
    
    properties(SetAccess = protected)
        % Network is a dlnetwork object.
        Network
        
        % AnchorBoxes is a N-by-1 cell array of M-by-2 array, where M is
        % number of anchor boxes, N is either network.Outputs or the
        % number of elements in DetectionNetworkSource.and M corresponds to
        % array of anchor boxes in [height width] format.
        AnchorBoxes
        
        % ClassNames specifies the names of object classes that the YOLO v3
        % object detector is pretrained or to be trained.
        ClassNames
        
        % An M-by-2 matrix defining the [height width] of image sizes used
        % to train the detector. During detection, an input image is
        % resized to nearest InputSize before it is processed by
        % the detection network when the DetectionPreprocessing NVP is 'auto'.
        InputSize
    end
    
    properties (Access = protected, Transient)
        FilterBboxesFunctor
    end
    
    properties(Access = private, Transient)
        % LayerIndices  A struct that caches indices to certain layers used
        %               frequently during detection.
        LayerIndices
    end
    
    properties(Dependent = true)
        Learnables
        State
    end
    
    methods
        function this = yolov3ObjectDetector(varargin)
            vision.internal.requiresNeuralToolbox(mfilename);
            narginchk(1,9);
            if (isa(varargin{1,1},'string')||isa(varargin{1,1},'char'))
                params = yolov3ObjectDetector.parsePretrainedDetectorInputs(varargin{:});
                this = vision.internal.cnn.yolov3(params);
                
                net = this.Network;
                lgraph = layerGraph(net);
                lgraph = iUpdateInputLayer(lgraph,params.InputSize);
                
                if ~isempty(params.DetectionNetworkSource)
                    numClasses = numel(params.ClassNames);
                    lgraph = iConfigureDetector(lgraph,numClasses,params.AnchorBoxes,params.DetectionNetworkSource);
                end
                
                this.Network = dlnetwork(lgraph);
            else
                % Codepath for dlnetwork, DAGNetwork, custom network
                % object.
                narginchk(3,9);
                params = yolov3ObjectDetector.parseDetectorInputs(varargin{:});
                this.Network = params.Network;
                this.ClassNames = categorical(params.ClassNames);
                this.AnchorBoxes = params.AnchorBoxes;
                this.InputSize = params.InputSize;
                this.ModelName = params.ModelName;
            end
            this.FilterBboxesFunctor = vision.internal.cnn.utils.FilterBboxesFunctor;
        end
        
        function val = get.Learnables(this)
            val = this.Network.Learnables;
        end
        
        function this = set.Learnables(this,val)
            this.Network.Learnables = val;
        end
        
        function val = get.State(this)
            val = this.Network.State;
        end
        
        function this = set.State(this,val)
            this.Network.State = val;
        end
        
        function this = set.Network(this, network)
            validateattributes(network, ...
                {'dlnetwork'},{'scalar'});
            
            % update layer index cache
            this = this.setLayerIndices(network);
            this.Network = network;
        end
        
        %------------------------------------------------------------------
        function varargout = preprocess(detector, I, varargin)
            %preprocess Preprocess input data.
            %
            % Use this method to preprocess the input data prior to calling
            % the predict or forward methods. This method can be used with
            % detector to create a datastore transform. The detect method
            % automatically preprocesses the input data when the
            % DetectionPreprocessing NVP is set to 'auto'.
            %
            % Ipreprocessed = preprocess(detector, I) resizes the input data
            % to the nearest detector.InputSize and scales the pixels to
            % between 0 and 1 when DetectionPreprocessing NVP of detect is
            % 'auto'. When DetectionPreprocessing is 'none', the output and
            % input are the same. The input detector is a
            % yolov3ObjectDetector object and I is an M-by-N-by-P image.
            %
            % data = preprocess(detector, data) resizes the input
            % training data to the nearest detector.InputSize and scales
            % the pixels to between 0 and 1, also updates the corresponding
            % bounding boxes to the resized image dimension. data
            % is a cell array containing image, bounding boxes, labels
            % within each row. bounding boxes is an M-by-4 array in
            % [x, y, width, height] format and are axis aligned. Image and
            % bounding boxes are resized to detector.InputSize by
            % preserving the width and height aspect ratio when
            % DetectionPreprocessing is 'auto'. When DetectionPreprocessing
            % is 'none', the output is the same as the input. detector is a
            % yolov3ObjectDetector object and data is a cell array.
            %
            % [...,info] = preprocess(detector, trainingData) optionally
            % returns the information related to resized image.
            % info is structure containing the following fields:
            %   PreprocessedImage   - Size of preprocessed image.
            %   ScaleX              - Scale factor used to resize the input
            %                         image in x direction.
            %   ScaleY              - Scale factor used to resize the input
            %                         image in y direction.
            %
            % Notes
            % -----
            % - When DetectionPreprocessing is 'auto', input I is
            %   resized to nearest detector.InputSize by preserving the
            %   width and height aspect ratio.
            %
            % Class Support
            % -------------
            % The input image I can be uint8, uint16, int16, double,
            % single, or logical, and it must be real and non-sparse.
            %
            % The output image Ipreprocessed is always single.
            %
            % Example: Preprocess input image.
            % --------------------------------
            % % Load pre-trained YOLO v3 detector.
            % detector = yolov3ObjectDetector('tiny-yolov3-coco');
            %
            % % Read test image.
            % I = imread('highway.png');
            %
            % % Preprocess input image.
            % Ipreprocess = preprocess(detector, I);
            %
            % % Display results.
            % montage({I, Ipreprocess});
            % title("Original Image (Left) vs. Preprocessed Image (Right)")
            %
            % See also yolov3ObjectDetector, imresize, rescale.
            
            if isempty(varargin)
                trainingImageSize = detector.InputSize;
                networkInputSize = detector.Network.Layers(1,1).InputSize;
                [varargout{1:nargout}] = iPreprocess(I, trainingImageSize, networkInputSize);  
            else
                params = parsePreprocessInputs(detector, I, varargin);
                if params.DetectionInputIsDatastore
                    % Copy and reset the given datastore, so external state events are
                    % not reflected.
                    ds = copy(I);
                    reset(ds);
                    
                    fcn = @iPreprocessForDetect;
                    % We need just the preprocessed image -> num arg out is 1.
                    fcnArgOut = 2;
                    varargout{1} = transform(ds, @(x)iPreProcessForDatastoreRead(x,fcn,fcnArgOut,...
                        params.DetectionPreprocessing,params.ROI,params.UseROI,detector.InputSize));
                    varargout{2} = {};
                else
                    [varargout{1:nargout}] = iPreprocessForDetect(I, ...
                        params.DetectionPreprocessing, params.ROI, params.UseROI, detector.InputSize);
                end
            end
        end
        
        %------------------------------------------------------------------
        function outputFeatures = predict(detector,dlX,varargin)
            % outputFeatures = predict(detector, dlX) predicts features of
            % the preprocessed image dlX. The outputFeatures is a N-by-8
            % cell array, where N are the number of outputs in
            % detector.Network. Each cell of outputFeature contains
            % predictions from an output layer. detector is a
            % yolov3ObjectDetector object and dlX is a SSCB formatted dlarray.
            %
            % Class Support
            % -------------
            % The input image dlX should be SSCB formatted dlarray,
            % real and non-sparse.
            %
            % Example
            % -------
            % % Load pre-trained YOLO v3 detector.
            % detector = yolov3ObjectDetector('tiny-yolov3-coco');
            %
            % % Read test image and convert to dlarray.
            % I = imread('highway.png');
            % I = single(rescale(I));
            % dlX = dlarray(I, 'SSCB');
            %
            % outputFeatures = predict(detector, dlX);
            %
            % See also yolov3ObjectDetector, dlarray, dlnetwork.
            
            % Compute network input size required for post processing.

            predictParams = parsePredictInputs(detector,dlX,varargin);
            network = detector.Network;
            anchorBoxes = detector.AnchorBoxes;
            
            if (~isnumeric(dlX) && ~ iscell(dlX))
                
                % Process datastore with network and output the predictions.
                loader = iCreateDataLoader(dlX,predictParams.MiniBatchSize,predictParams.NetworkInputSize);
                
                % Iterate through data and write results to disk.
                k = 1;
                
                bboxes = cell(predictParams.MiniBatchSize, 1);
                scores = cell(predictParams.MiniBatchSize, 1);
                labels = cell(predictParams.MiniBatchSize, 1);
                
                while hasdata(loader)
                    X = nextBatch(loader);
                    imgBatch = X{1};
                    batchInfo = X{2};
                    numMiniBatch = size(batchInfo,1);
                                        
                    % Compute predictions.
                    features = iPredictActivations(network, imgBatch, anchorBoxes);
                    
                    for ii = 1:numMiniBatch
                        fmap = cell(size(network.OutputNames'));
                        for i = 1:8
                            for j = 1:size(fmap,1)
                                fmap{j,i} = features{j,i}(:,:,:,ii);
                            end
                        end
                        [bboxes{k,1},scores{k,1},labels{k,1}] = ...
                            postprocess(detector,fmap, batchInfo{ii}, varargin{1,1});
                        k = k + 1;
                    end
                end
                varNames = {'Boxes', 'Scores', 'Labels'};
                outputFeatures = table(bboxes(1:k-1), scores(1:k-1), labels(1:k-1), 'VariableNames', varNames);
            else
                
                if iscell(dlX)
                    outputFeatures = iPredictBatchActivations(network, dlX, anchorBoxes);
                else
                    if size(dlX,4)>1
                        outputFeatures = iPredictMultiActivations(network, dlX, anchorBoxes);
                    else
                        outputFeatures = iPredictActivations(network, dlX, anchorBoxes);
                    end
                end
            end
        end
        
        %------------------------------------------------------------------
        function [features, act, state] = forward(detector,dlX)
            % [features, act] = forward(detector, dlX) computes 
            % features of the preprocessed image dlX. The features is
            % a N-by-8 cell arrray, where N are the number of outputs in
            % detector.Network. Each row of the features are the network 
            % activations converted from grid cell coordinates to box 
            % coordinates. The act is a N-by-8 cell array, where N are the 
            % number of outputs in detector.Network. Each row of act 
            % contains activations from an output layer. detector is a 
            % yolov3ObjectDetector object and dlX is a formatted dlarray.
            %
            % [..., state] = forward(detector, dlX) optionally return the
            % state information of the network. The state is used
            % to update the parameters of network in yolo object
            % detector.
            %
            % Class Support
            % -------------
            % The input image dlX should be formatted dlarray,
            % real and non-sparse.
            %
            % Example
            % -------
            % % Load pre-trained YOLO v3 detector.
            % detector = yolov3ObjectDetector('tiny-yolov3-coco');
            %
            % % Read test image and convert to dlarray.
            % I = imread('highway.png');
            % I = single(rescale(I));
            % dlX = dlarray(I, 'SSCB');
            %
            % outputFeatures = forward(detector, dlX);
            %
            % See also yolov3ObjectDetector, dlarray, dlnetwork.
            
            % Compute network input size required to compute loss.
            network = detector.Network;
            
            % Compute activations.
            act = cell(size(network.OutputNames'));
            [act{:},state] = forward(network, dlX);
            
            anchorBoxes = detector.AnchorBoxes;
            act = iYolov3Transform(act, anchorBoxes);
            
            % Gather the activations in the CPU for post processing and extract dlarray data.
            features = cellfun(@ gather, act,'UniformOutput',false);
            features = cellfun(@ extractdata, features, 'UniformOutput', false);
            
            % Convert predictions from grid cell coordinates to box coordinates.
            inputImageSize = size(dlX,1:2);
            features(:,2:5) = anchorBoxGenerator(detector,features(:,2:5),inputImageSize);
        end
        
        function varargout = detect(detector, I, varargin)
            % bboxes = detect(detector,I) detects objects within the image I.
            % The location of objects within I are returned in bboxes, an
            % M-by-4 matrix defining M bounding boxes. Each row of bboxes
            % contains a four-element vector, [x, y, width, height]. This
            % vector specifies the upper-left corner and size of a bounding
            % box in pixels. detector is a yolov3ObjectDetector object
            % and I is a truecolor or grayscale image.
            %
            % [..., scores] = detect(detector,I) optionally return the class
            % specific confidence scores for each bounding box. The scores
            % for each detection is product of objectness prediction and
            % classification scores. The range of the scores is [0 1].
            % Larger score values indicate higher confidence in the
            % detection.
            %
            % [..., labels] = detect(detector,I) optionally return the labels
            % assigned to the bounding boxes in an M-by-1 categorical
            % array. The labels used for object classes is defined during
            % training.
            %
            % detectionResults = detect(yolo,DS) detects objects within the
            % series of images returned by the read method of datastore,
            % DS. DS, must be a datastore that returns a table or a cell
            % array with the first column containing images.
            % detectionResults is a 3-column table with variable names
            % 'Boxes', 'Scores', and 'Labels' containing bounding boxes,
            % scores, and the labels. The location of objects within an
            % image, I are returned in bounding boxes, an M-by-4 matrix
            % defining M bounding boxes. Each row of boxes contains a
            % four-element vector, [x, y, width, height]. This vector
            % specifies the upper-left corner and size of a bounding box in
            % pixels. yolo is a yolov3ObjectDetector object.
            %
            % [...] = detect(..., roi) optionally detects objects within
            % the rectangular search region specified by roi. roi must be a
            % 4-element vector, [x, y, width, height], that defines a
            % rectangular region of interest fully contained in I. When
            % InputPreprocessing is set to 'none', roi cannot be specified.
            %
            % [...] = detect(..., Name, Value) specifies additional
            % name-value pairs described below:
            %
            % 'Threshold'              A scalar between 0 and 1. Detections
            %                          with scores less than the threshold
            %                          value are removed. Increase this value
            %                          to reduce false positives.
            %
            %                          Default: 0.5
            %
            % 'SelectStrongest'        A logical scalar. Set this to true to
            %                          eliminate overlapping bounding boxes
            %                          based on their scores. This process is
            %                          often referred to as non-maximum
            %                          suppression. Set this to false if you
            %                          want to perform a custom selection
            %                          operation. When set to false, all the
            %                          detected bounding boxes are returned.
            %
            %                          Default: true
            %
            % 'MinSize'                Specify the size of the smallest
            %                          region containing an object, in
            %                          pixels, as a two-element vector,
            %                          [height width]. When the minimum size
            %                          is known, you can reduce computation
            %                          time by setting this parameter to that
            %                          value. By default, 'MinSize' is the
            %                          smallest object that can be detected
            %                          by the trained network.
            %
            %                          Default: [1,1]
            %
            % 'MaxSize'                Specify the size of the biggest region
            %                          containing an object, in pixels, as a
            %                          two-element vector, [height width].
            %                          When the maximum object size is known,
            %                          you can reduce computation time by
            %                          setting this parameter to that value.
            %                          Otherwise, the maximum size is
            %                          determined based on the width and
            %                          height of I.
            %
            %                          Default: size(I)
            %
            % 'MiniBatchSize'          The mini-batch size used for processing a
            %                          large collection of images. Images are grouped
            %                          into mini-batches and processed as a batch to
            %                          improve computational efficiency. Larger
            %                          mini-batch sizes lead to faster processing, at
            %                          the cost of more memory.
            %
            %                          Default: 128
            %
            % 'DetectionPreprocessing' Specify whether or not the detect method
            %                          automatically preprocesses input images.
            %                          Valid values are 'auto' and 'none':
            %
            %                           'auto'      Resize the input data to the
            %                                       nearest InputSize by preserving the
            %                                       aspect ratio and rescales image
            %                                       pixels between 0 and 1.
            %
            %                           'none'      Input data is not
            %                                       preprocessed. If you choose 
            %                                       this option, the datatype 
            %                                       of the test image must be 
            %                                       either single or double.
            %
            %                         Default: 'auto'
            %
            %  Notes:
            %  -----
            %  - When 'SelectStrongest' is true the selectStrongestBboxMulticlass
            %    function is used to eliminate overlapping boxes. By
            %    default, the function is called as follows:
            %
            %   selectStrongestBboxMulticlass(bbox,scores,labels,...
            %                                       'RatioType', 'Union', ...
            %                                       'OverlapThreshold', 0.5);
            %
            %  - When the input image size does not match the network input size, the
            %    detector resizes the input image to the detector.InputSize.
            %
            %  - Input image is preprocessed by default when DetectionPreprocessing is 'auto'.
            %
            % Class Support
            % -------------
            % The input image I can be uint8, uint16, int16, double,
            % single, and it must be real and non-sparse.
            %
            % When DetectionPreprocessing is 'none', the input image I 
            % can be single or double, and it must be real and non-sparse.
            %
            % Example
            % -------
            % % Load pre-trained detector.
            % detector = yolov3ObjectDetector('tiny-yolov3-coco');
            %
            % % Read test image.
            % I = imread('highway.png');
            %
            % % Run detector.
            % [bboxes, scores, labels] = detect(detector, I);
            %
            % % Display results.
            % detectedImg = insertObjectAnnotation(I, 'Rectangle', bboxes, labels);
            % figure
            % imshow(detectedImg)
            %
            % See also yolov3ObjectDetector,
            % yolov3ObjectDetector/preprocess, yolov2ObjectDetector.
            
            params = parseDetectInputs(detector,I,varargin{:});
            [varargout{1:nargout}] = performDetect(detector, I, params);
        end
    end
    
    methods(Hidden)
        %------------------------------------------------------------------
        function params = parsePreprocessInputs(detector, I, varargin)
            % Preprocessing method during detect.
            params.DetectionPreprocessing = varargin{1,1}{1,1}.DetectionPreprocessing;
            if (~(strcmp(params.DetectionPreprocessing, 'auto')) && varargin{1,1}{1,1}.UseROI)
                error(message('yolov3:yolov3Detector:roiNotSupported'))
            end
            params.UseROI = varargin{1,1}{1,1}.UseROI;
            params.ROI = varargin{1,1}{1,1}.ROI;
            params.DetectionInputIsDatastore = ~isnumeric(I) && ~iscell(I);
        end
        
        %------------------------------------------------------------------        
        function params = parsePredictInputs(detector,dlX,varargin)
            if (isempty(varargin) || isempty(varargin{1,1}))
                params.MiniBatchSize = 1;
                params.NetworkInputSize = [];
                params.DetectionInputWasBatchOfImages = iscell(dlX);
            else
                params = varargin{1,1}{1,1};
            end
        end
        %------------------------------------------------------------------
        function params = parseDetectInputs(detector, I, varargin)
            
            params.DetectionInputIsDatastore = ~isnumeric(I);
            
            if params.DetectionInputIsDatastore
                sampleImage = vision.internal.cnn.validation.checkDetectionInputDatastore(I, mfilename);
            else
                sampleImage = I;
            end
            
            network = detector.Network;
            
            networkInputSize = network.Layers(detector.LayerIndices.InputLayerIdx).InputSize;
            
            validateChannelSize = true;  % check if the channel size is equal to that of the network
            validateImageSize   = false; % yolov3 can support images smaller than input size
            [sz,params.DetectionInputWasBatchOfImages] = vision.internal.cnn.validation.checkDetectionInputImage(...
                networkInputSize,sampleImage,validateChannelSize,validateImageSize);
            
            defaults = getDefaultYOLOv3DetectionParams();
            
            p = inputParser;
            p.addOptional('roi', defaults.roi);
            p.addParameter('SelectStrongest', defaults.SelectStrongest);
            p.addParameter('MinSize', defaults.MinSize);
            p.addParameter('MaxSize', sz(1:2));
            p.addParameter('MiniBatchSize', defaults.MiniBatchSize);
            p.addParameter('Threshold', defaults.Threshold);
            p.addParameter('DetectionPreprocessing', defaults.DetectionPreprocessing);
            parse(p, varargin{:});
            
            userInput = p.Results;
            
            vision.internal.cnn.validation.checkMiniBatchSize(userInput.MiniBatchSize, mfilename);
            
            useROI = ~ismember('roi', p.UsingDefaults);
            
            if useROI
                vision.internal.detector.checkROI(userInput.roi, sz);
            end
            
            vision.internal.inputValidation.validateLogical(...
                userInput.SelectStrongest, 'SelectStrongest');
            
            supportedInputs = ["auto", "none"];
            validatestring(userInput.DetectionPreprocessing, supportedInputs, mfilename, 'DetectionPreprocessing');
            
            % Validate minsize and maxsize.
            validateMinSize = ~ismember('MinSize', p.UsingDefaults);
            validateMaxSize = ~ismember('MaxSize', p.UsingDefaults);
            
            if validateMinSize
                vision.internal.detector.ValidationUtils.checkMinSize(userInput.MinSize, [1,1], mfilename);
            end
            
            if validateMaxSize
                vision.internal.detector.ValidationUtils.checkSize(userInput.MaxSize, 'MaxSize', mfilename);
                if useROI
                    coder.internal.errorIf(any(userInput.MaxSize > userInput.roi([4 3])) , ...
                        'vision:yolo:modelMaxSizeGTROISize',...
                        userInput.roi(1,4),userInput.roi(1,3));
                else
                    coder.internal.errorIf(any(userInput.MaxSize > sz(1:2)) , ...
                        'vision:yolo:modelMaxSizeGTImgSize',...
                        sz(1,1),sz(1,2));
                end
            end
            
            if validateMaxSize && validateMinSize
                coder.internal.errorIf(any(userInput.MinSize >= userInput.MaxSize) , ...
                    'vision:ObjectDetector:minSizeGTMaxSize');
            end
            
            if useROI
                if ~isempty(userInput.roi)
                    sz = userInput.roi([4 3]);
                    vision.internal.detector.ValidationUtils.checkImageSizes(sz(1:2), userInput, validateMinSize, ...
                        userInput.MinSize, ...
                        'vision:ObjectDetector:ROILessThanMinSize', ...
                        'vision:ObjectDetector:ROILessThanMinSize');
                end
            else
                vision.internal.detector.ValidationUtils.checkImageSizes(sz(1:2), userInput, validateMaxSize, ...
                    userInput.MinSize , ...
                    'vision:ObjectDetector:ImageLessThanMinSize', ...
                    'vision:ObjectDetector:ImageLessThanMinSize');
            end
            
            % Validate threshold.
            yolov3ObjectDetector.checkThreshold(userInput.Threshold);
            
            params.ROI                      = single(userInput.roi);
            params.UseROI                   = useROI;
            params.SelectStrongest          = logical(userInput.SelectStrongest);
            params.MinSize                  = single(userInput.MinSize);
            params.MaxSize                  = single(userInput.MaxSize);
            params.MiniBatchSize            = double(userInput.MiniBatchSize);
            params.Threshold                = single(userInput.Threshold);
            params.NetworkInputSize         = networkInputSize;
            params.FilterBboxesFunctor      = detector.FilterBboxesFunctor;
            params.DetectionPreprocessing   = userInput.DetectionPreprocessing;
        end
        
        %------------------------------------------------------------------
        function varargout = postprocess(detector,YPredData, info, params)
            if (istable(YPredData))
                [varargout{1:nargout}] = YPredData;
            else
                if params.DetectionInputWasBatchOfImages
                    [varargout{1:nargout}] = iPostprocessMultiDetection(detector,YPredData,info,params);
                else
                    [varargout{1:nargout}] = iPostprocessSingleDetection(detector,YPredData,info,params);
                end
            end
        end
        
        %--------------------------------------------------------------------------
        function tiledAnchors = anchorBoxGenerator(detector,YPredCell,inputImageSize)
            % Generate tiled anchor offset.
            anchorBoxes = detector.AnchorBoxes;
            tiledAnchors = cell(size(YPredCell));
            for i=1:size(YPredCell,1)
                anchors = anchorBoxes{i, :};
                [h,w,~,n] = size(YPredCell{i,1});
                [tiledAnchors{i,2}, tiledAnchors{i,1}] = ndgrid(0:h-1,0:w-1,1:size(anchors,1),1:n);
                [~,~,tiledAnchors{i,3}] = ndgrid(0:h-1,0:w-1,anchors(:,2),1:n);
                [~,~,tiledAnchors{i,4}] = ndgrid(0:h-1,0:w-1,anchors(:,1),1:n);
            end
            
            % Convert grid cell coordinates to box coordinates.
            for i=1:size(YPredCell,1)
                [h,w,~,~] = size(YPredCell{i,1});
                tiledAnchors{i,1} = (tiledAnchors{i,1}+YPredCell{i,1})./w;
                tiledAnchors{i,2} = (tiledAnchors{i,2}+YPredCell{i,2})./h;
                tiledAnchors{i,3} = (tiledAnchors{i,3}.*YPredCell{i,3})./inputImageSize(2);
                tiledAnchors{i,4} = (tiledAnchors{i,4}.*YPredCell{i,4})./inputImageSize(1);
            end
        end
        
        %------------------------------------------------------------------
        function this = setLayerIndices(this, network)
            this.LayerIndices.InputLayerIdx = yolov3ObjectDetector.findYOLOv3ImageInputLayer(network.Layers);
        end
    end
    
    methods(Static, Access = private)
        function params = parsePretrainedDetectorInputs(varargin)
            p = inputParser;
            if size(varargin,2) == 1
                p.addRequired('DetectorName');
                if strcmp(varargin{1,1},'darknet53-coco')
                    inpSz = [608,608];
                else
                    inpSz = [416,416];
                end
                p.addParameter('InputSize', inpSz);
                p.addParameter('DetectionNetworkSource',{});
                p.addParameter('ModelName', '', @iAssertValidLayerName);
                
                parse(p, varargin{:});
            else
                p.addRequired('DetectorName');
                p.addRequired('ClassNames');
                p.addRequired('AnchorBoxes');
                
                if strcmp(varargin{1,1},'darknet53-coco')
                    inpSz = [608,608];
                else
                    inpSz = [416,416];
                end
                p.addParameter('InputSize', inpSz);
                p.addParameter('DetectionNetworkSource',{});
                p.addParameter('ModelName', '', @iAssertValidLayerName);
                
                parse(p, varargin{:});
                
                params.ClassNames = p.Results.ClassNames(:);
                params.AnchorBoxes = p.Results.AnchorBoxes;
                
                iValidateClassNames(params.ClassNames);
                iValidateAnchorBoxes(params.AnchorBoxes);
            end
                params.InputSize = p.Results.InputSize;
                params.DetectionNetworkSource = p.Results.DetectionNetworkSource;
                params.ModelName = char(p.Results.ModelName);
                params.DetectorName = char(p.Results.DetectorName);
                
                supportedNetworks = ["darknet53-coco", "tiny-yolov3-coco"];
                validatestring(p.Results.DetectorName, supportedNetworks, mfilename, 'DetectorName', 1); 
                
                if strcmp(params.ModelName,'')
                    params.ModelName = p.Results.DetectorName;
                end

                iCheckInputSize(params.InputSize,[1,1]);
        end
        
        %------------------------------------------------------------------
        function params = parseDetectorInputs(varargin)
            validateattributes(varargin{1,1},{'DAGNetwork','dlnetwork'},...
                {'scalar'}, mfilename);
            p = inputParser;
            p.addRequired('Network');
            p.addRequired('ClassNames');
            p.addRequired('AnchorBoxes');
            
            networkImageSize = [];
            imgInputIdx = yolov3ObjectDetector.findYOLOv3ImageInputLayer(varargin{1,1}.Layers);
            
            if ~isempty(imgInputIdx)
                networkImageSize = varargin{1,1}.Layers(imgInputIdx,1).InputSize(1,1:2);
            end
            
            p.addParameter('InputSize', networkImageSize);
            p.addParameter('DetectionNetworkSource',{});
            p.addParameter('ModelName', '', @iAssertValidLayerName);
            parse(p, varargin{:});
            
            params.ClassNames = p.Results.ClassNames(:);
            params.AnchorBoxes = p.Results.AnchorBoxes;
            params.InputSize = p.Results.InputSize;
            params.DetectionNetworkSource = p.Results.DetectionNetworkSource;
            params.ModelName = char(p.Results.ModelName);
            
            iValidateClassNames(params.ClassNames);
            
            iValidateAnchorBoxes(params.AnchorBoxes);
            
            iCheckInputSize(params.InputSize,networkImageSize);
            
            net = p.Results.Network;
            
            numClasses = numel(params.ClassNames);
            
            lgraph = layerGraph(net);
            lgraph = iUpdateInputLayer(lgraph,params.InputSize);
            
            % Configure network for transfer Learning.
            if ~isempty(params.DetectionNetworkSource)
                % validate DetectionNetworkSources
                validateattributes(params.DetectionNetworkSource, {'cell','string'}, {'row', 'size', [1 NaN]}, ...
                    mfilename, 'DetectionNetworkSource');
                lgraph = iConfigureDetector(lgraph,numClasses,params.AnchorBoxes,params.DetectionNetworkSource);
                params.Network = dlnetwork(lgraph);
            else
                params.Network = dlnetwork(lgraph);
                iValidateYOLOv3Network(params.Network, numClasses, params.AnchorBoxes);
            end
        end
        
        %------------------------------------------------------------------
        function imageInputIdx = findYOLOv3ImageInputLayer(externalLayers)
            imageInputIdx = find(...
                arrayfun( @(x)isa(x,'nnet.cnn.layer.ImageInputLayer'), ...
                externalLayers));
        end
        %------------------------------------------------------------------
        % Validate Threshold value.
        %------------------------------------------------------------------
        function checkThreshold(threshold)
            validateattributes(threshold, {'single', 'double'}, {'nonempty', 'nonnan', ...
                'finite', 'nonsparse', 'real', 'scalar', '>=', 0, '<=', 1}, ...
                mfilename, 'Threshold');
        end
    end
    
    %======================================================================
    % Save/Load
    %======================================================================
    methods(Hidden)
        function s = saveobj(this)
            s.Version                  = 1.0;
            s.ModelName                = this.ModelName;
            s.Network                  = this.Network;
            s.ClassNames               = this.ClassNames;
            s.AnchorBoxes              = this.AnchorBoxes;
            s.InputSize                = this.InputSize;
        end
        
    end
    
    methods(Static, Hidden)
        function this = loadobj(s)
            try
                vision.internal.requiresNeuralToolbox(mfilename);
                network            = s.Network;
                classes            = s.ClassNames;
                anchorBoxes        = s.AnchorBoxes;
                inputSize          = s.InputSize;
                this = yolov3ObjectDetector(network,classes,anchorBoxes,...
                    'InputSize',inputSize);
                this.ModelName     = s.ModelName;
            catch ME
                rethrow(ME)
            end
        end
    end
    
    methods(Static, Hidden, Access = public)
        %------------------------------------------------------------------
        function data = preprocessInput(data, targetSize)
            % Resize image and boxes, then normalize image data between 0 and 1.
            if ~iscell(data)
                imgData = {data};
            else
                imgData = data;
            end
            
            for idx = 1:size(imgData,1)
                I = imgData{idx,1};
                I = single(rescale(I));
                if iscell(data)
                    [I,bboxes] = vision.internal.cnn.LetterBoxImage(I,targetSize(1:2),imgData{idx,2});
                    data(idx,1:2) = {I, bboxes};
                else
                    dataTmp = [];
                    for i = 1:size(I,4)
                        Itmp = I(:,:,:,i);
                        [Itmp,~] = vision.internal.cnn.LetterBoxImage(Itmp,targetSize(1:2));
                        if isempty(dataTmp)
                            dataTmp = Itmp;
                        else
                            dataTmp = cat(4,dataTmp,Itmp);
                        end
                    end
                    data = dataTmp;
                end
            end
        end
    end
end

%--------------------------------------------------------------------------
function lgraph = iRemoveLayers(lgraph,detectionNetSource)
% Remove all the layers after detectionNetworkSource.
dg = vision.internal.cnn.RCNNLayers.digraph(lgraph);

% Find the feature extraction nodes to remove nodes after feature
% extraction layer.
id = 1;
for i = 1:size(detectionNetSource,2)
    % Verify that all detectionNetworkSource layers exist in lgraph.
    iVerifyLayersExist(lgraph, detectionNetSource);
    nodeId = findnode(dg,char(detectionNetSource{1,i}));
    id = max(id,nodeId);
end

% Search for all nodes starting from the feature extraction
% layer.
if ~(sum(id)==0)
    ids = dfsearch(dg,id);
    names = dg.Nodes.Name(ids,:);
    lgraph = removeLayers(lgraph, names(2:end)); % exclude feature extraction layer which is first name.
end
end

%--------------------------------------------------------------------------
function iVerifyLayersExist(lgraph, layerNames)

numLayers = numel(lgraph.Layers);
for idx = 1:numel(layerNames)
    foundLayer = false;
    for lIdx = 1:numLayers
        if strcmp(layerNames{idx}, lgraph.Layers(lIdx).Name)
            foundLayer = true;
            break;
        end
    end
    if ~foundLayer
        error(message('vision:ssd:InvalidLayerName', layerNames{idx}));
    end
end
end

%--------------------------------------------------------------------------
function iCheckInputSize(inputSize,~)
validateattributes(inputSize, {'numeric'}, ...
    {'2d','ncols',2,'ndims',2,'nonempty','nonsparse',...
    'real','finite','integer','positive','nrows',1,});
end

%--------------------------------------------------------------------------
function iValidateClassNames(value)
if ~isvector(value) || ~iIsValidDataType(value)
    error(message('yolov3:yolov3Detector:invalidClasses'));
end
if iHasDuplicates(value)
    error(message('yolov3:yolov3Detector:duplicateClasses'));
end
if isempty(value)
    error(message('yolov3:yolov3Detector:invalidClasses'));
end
end

%--------------------------------------------------------------------------
function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

%--------------------------------------------------------------------------
function iValidateAnchorBoxes(value)
validateattributes(value, {'cell'},{'column','size',[NaN NaN]}, ...
    mfilename, 'AnchorBoxes');

for i = 1:size(value,1)
    validateattributes(value{i,1}, {'numeric'}, {'size', [NaN 2], 'real',...
        'nonnan', 'finite','positive'}, mfilename, 'AnchorBoxes');
end

end

function outputFeatures = iPredictMultiActivations(network,dlX, anchorBoxes)
numMiniBatch = size(dlX,4);
outputFeatures = cell(numMiniBatch,1);
for ii = 1:numMiniBatch
    inp = dlX(:,:,:,ii);
    outputFeatures{ii,1} = iPredictActivations(network, inp, anchorBoxes);
end
end

function outputFeatures = iPredictBatchActivations(network,dlX, anchorBoxes)
numMiniBatch = size(dlX,2);
outputFeatures = cell(numMiniBatch,1);
for ii = 1:numMiniBatch
    inp = dlX{ii};
    outputFeatures{ii,1} = iPredictActivations(network, inp, anchorBoxes);
end
end

function outputFeatures = iPredictActivations(network, dlX, anchorBoxes)
% Compute predictions.
features = cell(size(network.OutputNames'));
[features{:}] = predict(network, dlX);

outputFeatures = iYolov3Transform(features, anchorBoxes);
end

function [bboxes,scores,labels] = iPostprocessMultiDetection(detector,YPredData,info,params)
numMiniBatch = size(YPredData,1);
bboxes = cell(numMiniBatch, 1);
scores = cell(numMiniBatch, 1);
labels = cell(numMiniBatch, 1);
for ii = 1:numMiniBatch
    [bboxes{ii},scores{ii},labels{ii}] = ...
        iPostprocessSingleDetection (detector,YPredData{ii,1},info,params);
end
end

function [bboxes,scores,labels] = iPostprocessSingleDetection (detector,YPredData,info,params)
% Obtain the classnames detector is trained on.
classes = detector.ClassNames;

predictions = cellfun(@ gather, YPredData,'UniformOutput',false);
extractDetections = cellfun(@ extractdata, predictions, 'UniformOutput', false);

extractDetections(:,2:5) = anchorBoxGenerator(detector, extractDetections(:,2:5), params.NetworkInputSize);

% Apply following post processing steps to filter the detections:
% * Filter detections based on threshold.
% * Convert bboxes from spatial to pixel dimension.

% Combine the prediction from different heads.
detections(:,1:5) = cellfun(@ reshapePredictions,extractDetections(:,1:5), 'UniformOutput', false);
detections(:,6) = cellfun(@(a,b) reshapeClasses(a,b),extractDetections(:,6),repmat({numel(classes)}, size(extractDetections(:,6))),'UniformOutput', false);
detections = cell2mat(detections);

% Filter the classes based on (confidence score * class probability).
[classProbs, classIdx] = max(detections(:,6:end),[],2);
detections(:,1) = detections(:,1).*classProbs;
detections(:,6) = classIdx;

% Keep detections whose objectness score is greater than thresh.
detections = detections(detections(:,1)>=params.Threshold,:);

% [varargout{1:nargout}] = iPostProcessDetections(detections,classes,info,params);

[bboxes,scores,labels] = iPostProcessDetections(detections,classes,info,params);
end

%--------------------------------------------------------------------------
function [bboxes,scores,labels] = iPostProcessDetections(detections,classes,info,params)

if ~isempty(detections)
    
    scorePred = detections(:,1);
    bboxesTmp = detections(:,2:5);
    classPred = detections(:,6);
    
    if (strcmp(params.DetectionPreprocessing, 'auto'))
        
        % Obtain boxes for preprocesssed image.
        processedImageSize(2) = info.PreprocessedImageSize(2);
        processedImageSize(1) = info.PreprocessedImageSize(1);
        
        scale = [processedImageSize(2) processedImageSize(1) processedImageSize(2) processedImageSize(1)];
        bboxesTmp = bboxesTmp.*scale;
        
        % Convert x and y position of detections from centre to top-left.
        % Resize boxes to image size.
        bboxesTmp = iConvertCenterToTopLeft(bboxesTmp);
        
        % Resize boxes to original image size.
        inputImageSize(2) = info.ScaleX.*info.PreprocessedImageSize(2);
        inputImageSize(1) = info.ScaleY.*info.PreprocessedImageSize(1);
        
        [shiftedBboxes,shiftedImSz] = vision.internal.cnn.DeLetterBoxImage(bboxesTmp,info.PreprocessedImageSize,inputImageSize);
        bboxPred = iScaleBboxes(shiftedBboxes,inputImageSize,shiftedImSz);
        
    else
        inputImageSize(2) = info.ScaleX.*info.PreprocessedImageSize(2);
        inputImageSize(1) = info.ScaleY.*info.PreprocessedImageSize(1);
        
        scale = [inputImageSize(2) inputImageSize(1) inputImageSize(2) inputImageSize(1)];
        bboxPred = bboxesTmp.*scale;
        
        % Convert x and y position of detections from centre to top-left.
        % Resize boxes to image size.
        bboxPred = iConvertCenterToTopLeft(bboxPred);
    end
    
    % Filter boxes based on MinSize, MaxSize.
    [bboxPred, scorePred, classPred] = filterBBoxes(params.FilterBboxesFunctor,...
        params.MinSize,params.MaxSize,bboxPred,scorePred,classPred);
    
    % Apply NMS.
    if params.SelectStrongest
        [bboxes, scores, classNames] = selectStrongestBboxMulticlass(bboxPred, scorePred, classPred ,...
            'RatioType', 'Union', 'OverlapThreshold', 0.5);
    else
        bboxes = bboxPred;
        scores = scorePred;
        classNames = classPred;
    end
    
    % Apply ROI offset
    bboxes(:,1:2) = vision.internal.detector.addOffsetForROI(bboxes(:,1:2), params.ROI, params.UseROI);
    
    if (~isempty(bboxes) && (bboxes(1,1) + bboxes(1,3)) > inputImageSize (1,2))
        bboxes(1,3) = (inputImageSize (1,2) - bboxes(1,1));
    end
    
    if (~isempty(bboxes) &&(bboxes(1,2) + bboxes(1,4)) > inputImageSize (1,1))
        bboxes(1,4) = (inputImageSize (1,1) - bboxes(1,2));
    end
    
    % Convert classId to classNames.
    labels = categorical(classes);
    labels = labels(classNames);
    
else
    bboxes = zeros(0,4,'single');
    scores = zeros(0,1,'single');
    labels = categorical(cell(0,1),cellstr(classes));
end

end

%--------------------------------------------------------------------------
function iValidateYOLOv3Network(network, numClasses, anchorBoxes)
numOutputLayers = size(network.OutputNames,2);
layerNames = string({network.Layers.Name});

% Verfiy Fully Connected layer does not exist in the network.
iVerifyFullyConnectedExistence(network);

% Verfiy Global average pooling layer does not exist in the network.
iVerifyGlobalAvgPoolExistence(network);

for i = 1:numOutputLayers
    numAnchorsScale = size(anchorBoxes{i,1}, 1);
    numPredictorsPerAnchor = 5 + numClasses;
    
    expectedFilters = numAnchorsScale*numPredictorsPerAnchor;

    % Compute the number of filters for last convolution layer.
    layerIdx = find(strcmp(network.OutputNames{1,i}, layerNames));
    
    idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.Convolution2DLayer'),network.Layers(1:layerIdx));  
    filterIdx = find(idx,1,'last');

    actualFilters = network.Layers(filterIdx,1).NumFilters;
    
    if ~(expectedFilters == actualFilters)
        error(message('yolov3:yolov3Detector:invalidNumFilters',mat2str(expectedFilters),mat2str(numAnchorsScale),mat2str(numClasses)));
    end
    
end
end

%--------------------------------------------------------------------------
function tf = iIsValidDataType(value)
tf = iscategorical(value) || iscellstr(value) || isstring(value);
end

%--------------------------------------------------------------------------
function tf = iHasDuplicates(value)
tf = ~isequal(value, unique(value, 'stable'));
end

%--------------------------------------------------------------------------
function predictions = iYolov3Transform(YPredictions, anchorBoxes)

predictions = cell(size(YPredictions, 1),6);
for idx = 1:size(YPredictions, 1)
    % Get the required info on feature size.
    numChannelsPred = size(YPredictions{idx},3);
    numAnchors = size(anchorBoxes{idx},1);
    numPredElemsPerAnchors = numChannelsPred/numAnchors;
    channelsPredIdx = 1:numChannelsPred;
    predictionIdx = ones([1,numAnchors.*5]);
    
    % X positions.
    startIdx = 1;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,2} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];
    
    % Y positions.
    startIdx = 2;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,3} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];
    
    % Width.
    startIdx = 3;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,4} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];
    
    % Height.
    startIdx = 4;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,5} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];
    
    % Confidence scores.
    startIdx = 5;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,1} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];
    
    % Class probabilities.
    classIdx = setdiff(channelsPredIdx,predictionIdx);
    predictions{idx,6} = YPredictions{idx}(:,:,classIdx,:);
end

predictions(:,7:8) = predictions(:,4:5);

% Apply activation to the predicted cell array.
predictions(:,1:3) = cellfun(@ sigmoid ,predictions(:,1:3),'UniformOutput',false);
predictions(:,4:5) = cellfun(@ exp,predictions(:,4:5),'UniformOutput',false);
predictions(:,6) = cellfun(@ sigmoid ,predictions(:,6),'UniformOutput',false);
end

%--------------------------------------------------------------------------
% Convert x and y position of detections from centre to top-left.
function bboxes = iConvertCenterToTopLeft(bboxes)
bboxes(:,1) = bboxes(:,1)- bboxes(:,3)/2 + 0.5;
bboxes(:,2) = bboxes(:,2)- bboxes(:,4)/2 + 0.5;
bboxes = floor(bboxes);
bboxes(bboxes<1) = 1;
end

%--------------------------------------------------------------------------
function x = reshapePredictions(pred)
[h,w,c,n] = size(pred);
x = reshape(pred,h*w*c,1,n);
end

%--------------------------------------------------------------------------
function x = reshapeClasses(pred,numclasses)
[h,w,c,n] = size(pred);
numanchors = c/numclasses;
x = reshape(pred,h*w,numclasses,numanchors,n);
x = permute(x,[1,3,2,4]);
[h,w,c,n] = size(x);
x = reshape(x,h*w,c,n);
end

%--------------------------------------------------------------------------
function lgraph = iUpdateInputLayer(lgraph,imageSize)

imgIdx = arrayfun(@(x)isa(x,'nnet.cnn.layer.ImageInputLayer'),...
    lgraph.Layers);
imageInputIdx = find(imgIdx,1,'first');

idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.Convolution2DLayer'),...
    lgraph.Layers);
convIdx = find(idx,1,'first');
numChannels = lgraph.Layers(convIdx,1).NumChannels;

if (strcmp(numChannels,'auto'))
    sz = lgraph.Layers(imageInputIdx,1).InputSize;
    numChannels = sz(1,3);
end

% Replace Image input Layer
if size(imageSize,1)>1
    inputSize = [min(imageSize),numChannels];
else
    inputSize = [imageSize,numChannels];
end

imageInput = imageInputLayer(inputSize,'Name',lgraph.Layers(imageInputIdx).Name,'Normalization','none');

lgraph = replaceLayer(lgraph,lgraph.Layers(imageInputIdx).Name,...
    imageInput);

end

%--------------------------------------------------------------------------
function lgraph = iConfigureDetector(lgraph,numClasses,anchorBoxes,detectionNetworkSource)

% Create a layerGraph for transfer learning.
lgraph = iRemoveLayers(lgraph,detectionNetworkSource);

numOutputLayers = size(detectionNetworkSource,2);

numAnchorBoxGroups = size(anchorBoxes,1);
if numOutputLayers ~= numAnchorBoxGroups
    error(message('yolov3:yolov3Detector:numOutputLayerMismatch'));
end

analysis = nnet.internal.cnn.analyzer.NetworkAnalyzer(lgraph);

% Add detection heads to feature extraction layers.
numPredictorsPerAnchor = 5 + numClasses;
for idx = 1:numOutputLayers
    
    featureLayerIdx = arrayfun(@(x) x.Name == ...
        detectionNetworkSource{1,idx},analysis.LayerAnalyzers);
    
    % Verify that YOLO v3 DetectionNetworkSource output size is greater than [1,1].
    activationSize = analysis.LayerAnalyzers(featureLayerIdx).Outputs.Size{1,1};
    if (any(activationSize(1:2) < 2))
        error(message("vision:yolo:mustHaveValidFinalActivationsSize"));
    end
    
    outFilters = analysis.LayerAnalyzers(featureLayerIdx).Outputs.Size{1}(3);
    
    numAnchorsScale = size(anchorBoxes{idx,1}, 1);
    
    % Compute the number of filters for last convolution layer.
    numFilters = numAnchorsScale*numPredictorsPerAnchor;
     if mod(idx, 2) == 0
        firstConv = transposedConv2dLayer(3,outFilters,'Stride',2,'Name',['customConv',num2str(idx)],'WeightsInitializer','he', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
        %firstConv = convolution2dLayer(3,outFilters,'Padding','same','Name',['customConv',num2str(idx)],'WeightsInitializer','he');
    else
        firstConv = convolution2dLayer(3,(outFilters*2),'Padding','same','Name',['customConv',num2str(idx)],'WeightsInitializer','he', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    end
    
    detectionSubNetwork = [firstConv;
        batchNormalizationLayer('Name',['customBatchNorm',num2str(idx)],'ScaleLearnRateFactor', 10, 'OffsetLearnRateFactor', 10);
        reluLayer('Name',['customRelu',num2str(idx)])
        convolution2dLayer(1,numFilters,'Padding','same','Name',['customOutputConv',num2str(idx)],'WeightsInitializer','he', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
        ];
    
    % Add layers that concatenate the features.
    if mod(idx, 2) == 0
        featureGatherLayers = [
            convolution2dLayer(1,ceil(outFilters./2),'Padding','same','Name',['featureConv',num2str(idx)],'WeightsInitializer','he', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
            batchNormalizationLayer('Name',['featureBatchNorm',num2str(idx)],'ScaleLearnRateFactor', 10, 'OffsetLearnRateFactor', 10);
            reluLayer('Name',['featureRelu',num2str(idx)])];
            % resize2dLayer('EnableReferenceInput',true, 'Name',['featureResize',num2str(idx)]);
            % depthConcatenationLayer(2,'Name',['depthConcat',num2str(idx)]);
            
        detectionSubNetwork = [featureGatherLayers;detectionSubNetwork];
    end
    
    lgraph = addLayers(lgraph,detectionSubNetwork);
    
    % Connect appropriate layers.
    if mod(idx, 2) == 0
        lgraph = connectLayers(lgraph,detectionNetworkSource{1,idx-1},['featureConv',num2str(idx)]);
        %lgraph = connectLayers(lgraph,detectionNetworkSource{1,idx},['featureResize',num2str(idx),'/ref']);
       % layerToConnect = ['depthConcat',num2str(idx),'/in2'];
    else
        layerToConnect = ['customConv',num2str(idx)];
    end
     if mod(idx, 2) == 1
        lgraph = connectLayers(lgraph, detectionNetworkSource{1,idx}, layerToConnect);
     end
end
end

%--------------------------------------------------------------------------
function out = iPreProcessForDatastoreRead(in, fcn, numArgOut, varargin)
if isnumeric(in)
    % Numeric input
    in = {in};
end
if istable(in)
    % Table input
    in = in{:,1};
else
    % Cell input
    in = in(:,1);
end
numItems = numel(in);
out = cell(numItems, numArgOut);
for ii = 1:numel(in)
    [out{ii, 1:numArgOut}] = fcn(in{ii},varargin{:});
end
end

%--------------------------------------------------------------------------
function [Ipreprocessed,info] = iPreprocessForDetect(I, detectionPreprocessing, roi, useROI, trainingImageSize)

% Crop image if requested.
Iroi = vision.internal.detector.cropImageIfRequested(I, roi, useROI);

% Find the nearest training image size.
[info.PreprocessedImageSize,info.ScaleX,info.ScaleY] = iFindNearestTrainingImageSize(...
    size(Iroi),trainingImageSize);

if (strcmp(detectionPreprocessing, 'auto'))
    Ipreprocessed = yolov3ObjectDetector.preprocessInput(Iroi, info.PreprocessedImageSize);
else
    if ~(isa(Iroi,'double') || isa(Iroi,'single') || isa(Iroi,'gpuArray'))
        error(message('yolov3:yolov3Detector:unSupportedInputClass'));
    end
    Ipreprocessed = Iroi;
end

if (isa(Ipreprocessed,'double')||isa(Ipreprocessed,'single')||isa(Ipreprocessed,'gpuArray'))
    Ipreprocessed = dlarray(Ipreprocessed,'SSCB');
end
end

%--------------------------------------------------------------------------
function [dataPreprocessed,infoPreprocessed] = iPreprocess(data, trainingImageSize, networkInputSize)
% Resize image and boxes, then normalize image data between 0 and 1.
if iscell(data)
    dataPreprocessed = data;
    for idx = 1:size(data,1)
        I = data{idx,1};
        iValidateInput(networkInputSize, I);
        % Find the nearest training image size.
        [info.PreprocessedImageSize,info.ScaleX,info.ScaleY] = iFindNearestTrainingImageSize(...
            size(I),trainingImageSize);
        I = single(rescale(I));
        
        [I,bboxes] = vision.internal.cnn.LetterBoxImage(I,info.PreprocessedImageSize(1:2),data{idx,2});
        dataPreprocessed(idx,1:2) = {I, bboxes};
        infoPreprocessed(idx,1) = {info};
    end
else
    [info.PreprocessedImageSize,info.ScaleX,info.ScaleY] = iFindNearestTrainingImageSize(...
        size(data),trainingImageSize);    
    dataTmp = [];
    for i = 1:size(data,4)
        Itmp = data(:,:,:,i);
        iValidateInput(networkInputSize, Itmp);
        [Itmp,~] = vision.internal.cnn.LetterBoxImage(Itmp,info.PreprocessedImageSize(1:2));
        if isempty(dataTmp)
            dataTmp = Itmp;
        else
            dataTmp = cat(4,dataTmp,Itmp);
        end
    end
    dataPreprocessed = dataTmp;
    infoPreprocessed = info;
end

end

%--------------------------------------------------------------------------
function iValidateInput(networkInputSize, sampleImage)
validateChannelSize = true;  % check if the channel size is equal to that of the network
validateImageSize   = false; % yolov3 can support images smaller than input size
[~, params.DetectionInputWasBatchOfImages] = vision.internal.cnn.validation.checkDetectionInputImage(...
    networkInputSize,sampleImage,validateChannelSize,validateImageSize);
end

%--------------------------------------------------------------------------
function loader = iCreateDataLoader(ds,miniBatchSize,inputLayerSize)
loader = nnet.internal.cnn.DataLoader(ds,...
    'MiniBatchSize',miniBatchSize,...
    'CollateFcn',@(x)iTryToBatchData(x,inputLayerSize));
end

%--------------------------------------------------------------------------
function data = iTryToBatchData(X, inputLayerSize)
try
    observationDim = numel(inputLayerSize) + 1;
    data{1} = cat(observationDim,X{:,1});
catch e
    if strcmp(e.identifier, 'MATLAB:catenate:dimensionMismatch')
        error(message('vision:ObjectDetector:unableToBatchImagesForDetect'));
    else
        throwAsCaller(e);
    end
end
data{2} = X(:,2:end);
end

%--------------------------------------------------------------------------
function s = getDefaultYOLOv3DetectionParams()
s.roi                     = zeros(0,4);
s.SelectStrongest         = true;
s.Threshold               = 0.5;
s.MinSize                 = [1,1];
s.MaxSize                 = [];
s.MiniBatchSize           = 128;
s.DetectionPreprocessing  = 'auto';
end

%--------------------------------------------------------------------------
function bboxPred = iScaleBboxes(bboxes,imSz,newImSz)
scale   = imSz(1:2)./newImSz;
[info.ScaleX,info.ScaleY] = deal(scale(2),scale(1));

bboxesX1Y1X2Y2 = vision.internal.cnn.boxUtils.xywhToX1Y1X2Y2(bboxes);
% Saturate X2,Y2 to the original image dimension, to remove the gray scale
% scale detections if any.
bboxesX1Y1X2Y2(:,3) = min(bboxesX1Y1X2Y2(:,3),newImSz(1,2));
bboxesX1Y1X2Y2(:,4) = min(bboxesX1Y1X2Y2(:,4),newImSz(1,1));

% Scale the boxes to the image dimension.
bboxesX1Y1X2Y2 = vision.internal.cnn.boxUtils.scaleX1X2Y1Y2(bboxesX1Y1X2Y2, info.ScaleX, info.ScaleY);
bboxPred = vision.internal.cnn.boxUtils.x1y1x2y2ToXYWH(bboxesX1Y1X2Y2);
end

%--------------------------------------------------------------------------
function [targetSize, sx, sy] = iFindNearestTrainingImageSize(sz,trainingImageSize)
idx = iComputeBestMatch(sz(1:2),trainingImageSize);
targetSize = trainingImageSize(idx,:);

% Compute scale factors to scale boxes from targetSize back to the input
% size.
scale   = sz(1:2)./targetSize;
[sx,sy] = deal(scale(2),scale(1));
end

%--------------------------------------------------------------------------
% Get the index of nearest size in TrainingImageSize training sizes that
% matches given image.
%--------------------------------------------------------------------------
function ind = iComputeBestMatch(preprocessedImageSize,trainingImageSize)
preprocessedImageSize = repmat(preprocessedImageSize,size(trainingImageSize,1),1);
Xdist = (preprocessedImageSize(:,1) - trainingImageSize(:,1));
Ydist = (preprocessedImageSize(:,2) - trainingImageSize(:,2));
dist = sqrt(Xdist.^2 + Ydist.^2);
[~,ind] = min(dist);
end

%--------------------------------------------------------------------------
function iVerifyFullyConnectedExistence(network)
% YOLOv3 network is based on Convolution Layers and should not
% contain any fullyConnected Layers.
idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.FullyConnectedLayer'),...
    network.Layers);    
if sum(idx) ~= 0
    error(message("vision:yolo:mustNotHaveAnyFCLayer"));
end
end

%--------------------------------------------------------------------------
function iVerifyGlobalAvgPoolExistence(network)
% YOLOv3 network should not contain any global average pooling layer as it
% downsamples input feature map to size of [1,1].
idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.GlobalAveragePooling2DLayer')||isa(x,'nnet.cnn.layer.GlobalMaxPooling2DLayer'),...
    network.Layers);    
if sum(idx) ~= 0
    error(message("vision:yolo:mustNotHaveAnyGlobalPoolingLayer"));
end
end
