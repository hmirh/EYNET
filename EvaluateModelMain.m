function [res] = EvaluateModelMain(results,testData, inputSize, groundTruthBboxes, name, validationphase)
if validationphase == 1
    detectionThreshold = 0.5;
    res = EvaluateModel(results, testData, inputSize, detectionThreshold, '', validationphase);
    return;
else 
    res = -1;
end
disp('-------------------------------------------');
detectionThreshold = 0.5;
disp(num2str(EvaluateModel(results, testData, inputSize, detectionThreshold, strcat(name, '_50' ), validationphase)));
disp('-------------------------------------------');
detectionThreshold = 0.75;
disp(num2str(EvaluateModel(results, testData, inputSize, detectionThreshold, strcat(name, '_75' ), validationphase)));
disp('-------------------------------------------');
% -------------------------------------------------------------------------
step = 0.5:0.05:0.95;
ap = zeros(1, 10);
for i = 1 : 10
    detectionThreshold =  step(i);
    ap(1, i) = EvaluateModel(results, testData, inputSize, detectionThreshold, '', validationphase);
end
disp(strcat('AP:',num2str(mean(ap))));
disp('-------------------------------------------');
% -------------------------------------------------------------------------
Temp = cell2mat(groundTruthBboxes);
area = Temp(:,3) .* Temp(:,4);
[val, ~] = sort(area);
step = (val(end) - val(1))/3;
small = (area<step);
med = (area>=step & area<2*step);
large = (area>=2*step);
disp('-------------------------------------------');
% -------------------------------------------------------------------------
detectionThreshold = 0.5;
Temp = testData;
Tempresults = results;
Tempresults(small == 0, :) = [];
Temp = subset(Temp, small == 1);
disp(num2str(EvaluateModel(Tempresults, Temp, inputSize, detectionThreshold, '', validationphase)));
disp('-------------------------------------------');
% -------------------------------------------------------------------------
Temp = testData;
Tempresults = results;
Tempresults(med == 0, :) = [];
Temp = subset(Temp, med == 1);
disp(num2str(EvaluateModel(Tempresults, Temp, inputSize, detectionThreshold, '', validationphase)));
disp('-------------------------------------------');
% -------------------------------------------------------------------------
Temp = testData;
Tempresults = results;
Tempresults(large == 0, :) = [];
Temp = subset(Temp, large == 1);
disp(num2str(EvaluateModel(Tempresults, Temp, inputSize, detectionThreshold, '', validationphase)));
% -------------------------------------------------------------------------
end