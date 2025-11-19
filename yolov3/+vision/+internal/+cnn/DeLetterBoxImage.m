function [bboxes,newImSz] = DeLetterBoxImage(bboxes,inpSz,imgSz)
% DeLetterBoxImage returns the shifted box coordinates by removing the
% gray canvas that is added in LetterBoxImage. The new box coordinates
% are with respect to the resized image obtained before applying the gray
% canvas.
%
% Input bboxes are the bounding boxes in [x,y,w,h] format, computed using 
% the letter boxed image.
%
% Input inpSz is the input image size of the network.
%
% Input imgSz is the original image size.

% Copyright 2021 The MathWorks, Inc

arI = imgSz(1,1)./imgSz(1,2);
if arI<1
    IcolFin = inpSz(1,2);
    IrowFin = IcolFin.*arI;
    
    % Compute the canvas shift.
    buff    = inpSz(1,1)-IrowFin;
    dcShift = buff/2;
    
    % Update the boxes based on the shift.
    bboxes(:,2) = max(1,(bboxes(:,2) - dcShift));
    newImSz = [IrowFin,IcolFin];
else
    IrowFin = inpSz(1,1);
    IcolFin = IrowFin./arI;
    
    % Compute the canvas shift.
    buff    = inpSz(1,2)-IcolFin;
    dcShift = buff/2;
    
    % Update the boxes based on the shift.
    bboxes(:,1) = max(1,(bboxes(:,1) - dcShift));
    newImSz = [IrowFin,IcolFin];
end
end