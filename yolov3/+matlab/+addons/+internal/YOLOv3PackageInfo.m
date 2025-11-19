classdef YOLOv3PackageInfo < matlab.addons.internal.SupportPackageInfoBase
    %YOLOv3PackageInfo MATLAB Compiler support information
    
    %   Copyright 2021 The MathWorks, Inc.
    
    methods
        function obj = YOLOv3PackageInfo()
            obj.baseProduct = 'Computer Vision Toolbox';
            obj.displayName = 'Computer Vision Toolbox Model for YOLO v3 Object Detection';
            obj.name        = 'Computer Vision Toolbox Model for YOLO v3 Object Detection';
            
            sproot = matlabshared.supportpkg.getSupportPackageRoot();
            
            % Define all the data that should be deployed from the support
            % package. This includes the actual language data, which will
            % be archived in the CTF.
            obj.mandatoryIncludeList = {...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3') ...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3','+vision') ...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3','license_addendum') ...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3','data','yolov3COCO.mat') ...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3','data','tinyYOLOv3COCO.mat') ...
                fullfile(sproot, 'toolbox','vision','supportpackages','yolov3','resources') ...
                fullfile(sproot, 'resources','yolov3')
                };
        end
    end
end