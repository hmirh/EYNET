function [Inew,bboxnew] = LetterBoxImage(I,targetSize,varargin)
% LetterBoxImage returns a resized image by preserving the width and height
% aspect ratio of input Image I. targetSize is a 1-by-2 vector consisting 
% the target dimension.
%
% Input I can be uint8, uint16, int16, double, single, or logical, and must
% be real and non-sparse.

% Copyright 2021 The MathWorks, Inc.

[Irow,Icol,Ichannels] = size(I);
bboxnew = [];

% Compute aspect Ratio.
arI = Irow./Icol;

% Preserve the maximum dimension based on the aspect ratio.
if arI<1
    IcolFin = targetSize(1,2);
    IrowFin = floor(IcolFin.*arI);
else
    IrowFin = targetSize(1,1);
    IcolFin = floor(IrowFin./arI);
end

% Resize the input image.
Itmp = imresize(I,[IrowFin,IcolFin]);

% Initialize Inew with gray values.
Inew = ones([targetSize,Ichannels],'like',I).*0.5;

% Compute the offset.
if arI<1
    buff = targetSize(1,1)-IrowFin;
else
    buff = targetSize(1,2)-IcolFin;
end

% Place the resized image on the canvas image.
if (buff==0)
    Inew = Itmp;
    if ~isempty(varargin)
        imgSize = size(I,1:2);
        boxScale = targetSize./imgSize;
        bboxnew = bboxresize(varargin{1,1},boxScale);
    end
else
    % When buffVal <=1, leave out the last row/column by starting from 1.
    buffVal = max(floor(buff/2),1);
    if arI<1
        Inew(buffVal:buffVal+IrowFin-1,:,:) = Itmp;
        if ~isempty(varargin)
            % Resize bounding boxes.
            bboxnew = iScaleBboxes(varargin{1,1},size(Itmp),size(I));
            bboxnew(:,2) = bboxnew(:,2)+buffVal;
        end
    else
        Inew(:,buffVal:buffVal+IcolFin-1,:) = Itmp;
        if ~isempty(varargin)
            % Resize bounding boxes.
            bboxnew = iScaleBboxes(varargin{1,1},size(Itmp),size(I));
            bboxnew(:,1) = bboxnew(:,1)+buffVal;
        end
    end
end

end


%--------------------------------------------------------------------------
function bboxPred = iScaleBboxes(bboxes,imSz,newImSz)
scale   = imSz(1:2)./newImSz(1:2);
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