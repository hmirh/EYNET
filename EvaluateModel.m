function [res] = EvaluateModel(results, testData, inputSize, detectionThreshold, name, validationphase)

groundTruthBboxes = preprocessDatagroundTruthBboxes(testData.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 2}.LabelData(:,1), [800 800], inputSize);
[precision,  recall] = bboxPrecisionRecall(cell2table(results{:,1}), ...
    table(groundTruthBboxes), ...
    detectionThreshold);
F1score = (precision * recall)/((precision + recall)/2);
% ------------------------------------------------------------------------
%DR = ((size(results, 1) - sum(cellfun(@(x) isempty(x),results{:, 2})))  / size(results, 1)) * 100;
% ------------------------------------------------------------------------
tp = 0;fp = 0;fn = 0;
for i = 1: size(results, 1)     
   if(bboxOverlapRatio(cell2mat(table2cell(results(i, 1))), ...
        cell2mat(groundTruthBboxes(i, 1)))) >= detectionThreshold
           tp = tp + 1;
   else
       if isempty(cell2mat(table2cell(results(i, 1))))
           fn = fn + 1;           
       else
           fp = fp + 1;
       end
   end
end
DR = tp / (tp + fp);
ACC = tp / (tp + fp + fn);
disp(strcat('F1score:', num2str(F1score), ', Recall:', num2str(recall), ', Precision:', num2str(precision), ...
    ', Detection Rate:', num2str(DR* 100), ', Accuracy:', num2str(ACC* 100)));
% ------------------------------------------------------------------------
[ap, recalll, precisionn] = evaluateDetectionPrecision(results, testData, detectionThreshold);
[logAverageMissRate, fppi, missRate] = evaluateDetectionMissRate(results, testData, detectionThreshold);
if validationphase == 1
    res.ap = ap;
    res.recall = recall;
    res.precision = precision; 
    res.F1score = F1score;
    res.ACC = ACC;
    res.logAverageMissRate = logAverageMissRate;
    return;
else
    res = ap;
end
% ------------------------------------------------------------------------
figure('rend','painters','pos',[10 10 1500 800]);
subplot(1,2,1);
plot(recalll, precisionn, 'Color','r', 'LineWidth',3)
xlabel('Recall')
ylabel('Precision')
grid on;
set(gca, 'FontSize', 12, 'FontWeight','bold','color',[0.95 0.95 0.95]);
title(sprintf('Average Precision = %.2f', ap));
% ------------------------------------------------------------------------
subplot(1,2,2);
loglog(fppi, missRate, 'Color','r', 'LineWidth',3);
grid on
set(gca, 'FontSize', 12, 'FontWeight','bold','color',[0.95 0.95 0.95]);
title(sprintf('Log Average Miss Rate = %.1f', logAverageMissRate))
if name ~= ""    
    save(strcat('Plot\recall', name, '.mat'), 'recalll');
    save(strcat('Plot\precision',name, '.mat'), 'precisionn');
    save(strcat('Plot\fppi', name, '.mat'), 'fppi');
    save(strcat('Plot\missRate', name, '.mat'), 'missRate');
end
% ------------------------------------------------------------------------
end