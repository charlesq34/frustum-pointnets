function [apScore,precision,reccall,isTp,isFp,isMissed,gtAssignment,maxOverlaps,numOfgt,gtIdxAll,allOverlaps] ...
    = computePRCurveTightBB(classname,predictedBbsTight,imageIds,groundTruthBbs,isDifficult)
    if length(groundTruthBbs) ~= length(isDifficult), error('inconsistent difficulty size.'); end
    if size(groundTruthBbs,1) ==1,groundTruthBbs= groundTruthBbs';end
    if size(isDifficult,1) ==1,isDifficult= isDifficult';end
    
    gtAssignment = zeros(length(predictedBbsTight),1);
    
    % pick ground truth of current class
    isMissed = false(numel(groundTruthBbs),1);
    isSameClass = ismember({groundTruthBbs.classname},classname);
    numOfgt =sum(isSameClass);
    groundTruthBbs = groundTruthBbs(isSameClass);
    isDifficult = logical(isDifficult(isSameClass));

    P = size(predictedBbsTight,1);
    G = numel(groundTruthBbs);
    
    % sort all detections
    [~,sortIdx] = sort([predictedBbsTight.confidence],'descend');
    %predictedBbs = predictedBbs(sortIdx,:);
    predictedBbsTight =predictedBbsTight(sortIdx,:);
    imageIds = imageIds(sortIdx);
    % threshold overlap
    tic;
    %predictedBbs =predictedBbs(:,1:6);
    %allOverlaps = bb3dOverlapCloseForm(predictedBbs,groundTruthBbs);
    allOverlaps = bb3dOverlapCloseForm(predictedBbsTight,groundTruthBbs);
    toc;
    onSameImage = bsxfun(@eq,imageIds(:),[groundTruthBbs.imageNum]);
    allOverlaps(~onSameImage) = 0;
    [maxOverlaps,gtIdx] = max(allOverlaps,[],2);
    gtIdx(maxOverlaps<eps) = 0;
    gtIdxAll = gtIdx;
    isOverlapping = maxOverlaps >= 0.25;
    gtIdx(~isOverlapping) = 0;
    
    % Assign ground truth to the best matched detection
    [uniqueGtIdx,firstAssignment,~] = unique(gtIdx,'first');
    isFirstAssignment = false(P,1);
    isFirstAssignment(firstAssignment) = true;
    isFirstAssignment = isFirstAssignment & gtIdx>0;
    
    % Get GT assignment of each TP detection
    tmpGtAssignment = zeros(P,1);
    tmpGtAssignment(firstAssignment) = uniqueGtIdx;
    tmp = find(isSameClass);
    tmpGtAssignment(tmpGtAssignment>0) = tmp(tmpGtAssignment(tmpGtAssignment>0));
    gtAssignment(sortIdx) = tmpGtAssignment;
    
%     % Assign best matched conf to ground truth
%     tmpGtConf = -1e10 * ones(G,1);
%     if uniqueGtIdx(1) == 0,
%         tmpGtConf(uniqueGtIdx(2:end)) = predictedBbs(firstAssignment(2:end),7);
%     else
%         tmpGtConf(uniqueGtIdx) = predictedBbs(firstAssignment,7);
%     end
%     gtConf(isSameClass) = tmpGtConf;
%     isMissed = gtConf < lowConfThresh;
    
    % get unassigned ground truth bounding boxes
    tmpIsMissed = true(G,1);
    tmpIsMissed(setdiff(uniqueGtIdx,0)) = false;
    isMissed(isSameClass) = tmpIsMissed;
    
    % assign detection labels
    tp = isFirstAssignment & isOverlapping;
    fp = ~tp;
    dc = gtIdx ~= 0 & isDifficult(max(1,gtIdx));
    tp(dc) = false;
    fp(dc) = false;
    isTp = false(numel(tp),1);
    isTp(sortIdx) = tp;
    isFp = false(numel(fp),1);
    isFp(sortIdx) = fp;
    
    % Compute precision/recall
    sumFp = cumsum(double(fp));
    sumTp = cumsum(double(tp));
    reccall = sumTp / sum(~isDifficult);
    precision = sumTp ./ (sumFp + sumTp);

    apScore = get_average_precision(precision, reccall);
end