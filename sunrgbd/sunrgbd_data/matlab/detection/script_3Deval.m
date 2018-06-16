% The result file should contains following feild
% allTestImgIds: Nx1 array of testing image's id in "alltest" for each box
% allBb3dtight : Nx1 cell 3D bounding box strcture 
clear all
toolboxpath = '/afs/cs.stanford.edu/u/rqi/Data/SUNRGBD/SUNRGBDtoolbox';
addpath(genpath(toolboxpath));


%load('./chair_demo.mat','allTestImgIds','allBb3dtight')
className ='chair';
load('./chair_demo.mat','allTestImgIds','allBb3dtight')
%load('../exampleresult_bathtub.mat');
%className ='bathtub';
split = load(fullfile(toolboxpath,'/traintestSUNRGBD/allsplit.mat'));
testset_path = split.alltest;    

for i = 1:length(testset_path)
    testset_path{i}(1:16) = '';
    testset_path{i} = strcat('/data/rqi/SUNRGBD', testset_path{i});
end
[groundTruthBbs,all_sequenceName] = benchmark_groundtruth(className,fullfile(toolboxpath,'Metadata/'),testset_path);
[apScore,precision,recall,isTp,isFp,isMissed,gtAssignment,maxOverlaps] = computePRCurve3D(className,allBb3dtight,allTestImgIds,groundTruthBbs,zeros(length(groundTruthBbs),1));
result_all = struct('apScore',apScore,'precision',precision,'recall',recall,'isTp',isTp,'isFp',isFp,'isMissed',isMissed,'gtAssignment',gtAssignment);

figure,
plot(recall,precision)
title(className)