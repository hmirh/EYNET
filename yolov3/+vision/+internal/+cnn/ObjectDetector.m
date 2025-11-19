classdef (Abstract) ObjectDetector < vision.internal.EnforceScalarValue 
    % ObjectDetector   Interface for Object Detectors.
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties
        % ModelName (char vector)   A name for the layer
        ModelName = '';
    end
    
    methods (Abstract)
        % predict   Forward preprocessed input data through the model 
        %           and output the result.
        %
        % Syntax
        %   varargout = predict( aDetector, X)
        %
        % Inputs
        %   aDetector - the detector to forward through.
        %   X         - the input to forward propagate through the
        %               model.
        %
        % Output
        %   varargout - the outputs of forward propagation through the 
        %               model.
        varargout = predict( aDetector, varargin )

        % preprocess   Preprocess input data to the required format.
        %
        % Syntax
        %   varargout = preprocess( aDetector, X )
        %
        % Inputs
        %   aDetector - the detector to forward through
        %   X         - the input to be preprocessed.
        %
        % Output
        %   varargout - the output of preprocessing function.        
        varargout = preprocess( aDetector, varargin )

        % postprocess   Postprocess function that converts predicted 
        %               features to detections.
        %
        % Syntax
        %   varargout = postProcessingFcn( aDetector, features )
        %
        % Inputs
        %   aDetector - the detector to forward through
        %   features  - the features to be postprocessed.
        %
        % Output
        %   varargout - the output of postprocessing function.         
        varargout = postprocess( aDetector, varargin )
    end
    
    
    methods(Hidden, Access = protected)
        
        function varargout = performDetect(this, I, params)
            [Ipreprocessed, info]  = this.preprocess(I,params);
            features = this.predict(Ipreprocessed,params);
            [varargout{1:nargout}] = this.postprocess(features,info,params);
        end
    end
    
    methods
        function layer = set.ModelName( layer, val )
            iEvalAndThrow(@()iAssertValidDetectorName( val ));
            layer.ModelName = convertStringsToChars( val );
        end
    end
    
    methods (Static, Hidden, Access = public)
        function name = matlabCodegenRedirect(~)
            % vision.internal.cnn.ObjectDetector is not supported for
            % codegen vision.internal.cnn.coder.ObjectDetector is the
            % implementation of vision.internal.cnn.ObjectDetector that
            % supports codegen for inference detect when generating code
            % for any custom layer inheriting from
            % vision.internal.cnn.ObjectDetector class, we will redirect it
            % to the codegenable implementation
            %
            % any changes made in vision.vision.Detector that affects the
            % detect call needs to be implemented in
            % vision.internal.cnn.coder.ObjectDetector as well
            name = 'ObjectCodegenDetector';
        end
    end
end

function iAssertValidDetectorName( name )
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName( name ));
end

function varargout = iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    [varargout{1:nargout}] = func();
catch exception
    throwAsCaller(exception)
end
end
