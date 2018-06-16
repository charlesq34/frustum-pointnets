%% Extract GT boxes

clear all
toolboxpath = '/afs/cs.stanford.edu/u/rqi/Data/SUNRGBD/SUNRGBDtoolbox';
addpath(genpath(toolboxpath));


%load('./chair_demo.mat','allTestImgIds','allBb3dtight')
for className = {'bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'}
    clear('bbs');
    clear('imgids');
    split = load(fullfile(toolboxpath,'/traintestSUNRGBD/allsplit.mat'));
    testset_path = split.alltest;    

    for i = 1:length(testset_path)
        testset_path{i}(1:16) = '';
        testset_path{i} = strcat('/data/rqi/SUNRGBD', testset_path{i});
    end
    [groundTruthBbs,all_sequenceName] = benchmark_groundtruth(className,fullfile(toolboxpath,'Metadata/'),testset_path);

    nBb = length(groundTruthBbs);
    for i = 1:nBb
        corners = get_corners_of_bb3d(groundTruthBbs(i));
        bbs(i,:) = [reshape([corners(1:4,1) corners(1:4,2)]',1,[]) min(corners([1 end],3)) max(corners([1 end],3))];
        imgids(i) = groundTruthBbs(i).imageNum;
    end

    dlmwrite(strcat(className{1}, '_gt_boxes.dat'), bbs, 'delimiter', ' ');
    dlmwrite(strcat(className{1}, '_gt_imgids.txt'), imgids, 'delimiter', ' ');
end
