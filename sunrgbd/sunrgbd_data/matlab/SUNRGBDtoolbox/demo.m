addpath(genpath('.'))
load('./Metadata/SUNRGBDMeta.mat')
load('./Metadata/SUNRGBD2Dseg.mat')
%% Read
imageId = 1;
data = SUNRGBDMeta(imageId);
% data.depthpath = '/data/rqi/SUNRGBD/SUNRGBD/kv2/kinect2data/000037_2014-05-26_14-54-02_260595134347_rgbf000041-resize/depth/0000041.png';
% data.rgbpath = '/data/rqi/SUNRGBD/SUNRGBD/kv2/kinect2data/000037_2014-05-26_14-54-02_260595134347_rgbf000041-resize/image/0000041.jpg';
data.depthpath(1:16) = '';
data.depthpath = strcat('/data/rqi/SUNRGBD',data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('/data/rqi/SUNRGBD',data.rgbpath);
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
%% draw 
figure,
imshow(data.rgbpath);
hold on; 
for kk =1:length(data.groundtruth3DBB)
    rectangle('Position', [data.groundtruth3DBB(kk).gtBb2D(1) data.groundtruth3DBB(kk).gtBb2D(2) data.groundtruth3DBB(kk).gtBb2D(3) data.groundtruth3DBB(kk).gtBb2D(4)],'edgecolor','y');
    text(data.groundtruth3DBB(kk).gtBb2D(1),data.groundtruth3DBB(kk).gtBb2D(2),data.groundtruth3DBB(kk).classname,'BackgroundColor','y')
end
%% draw 3D 
figure,
vis_point_cloud(points3d,rgb)
hold on;
for kk =1:length(data.groundtruth3DBB)
   vis_cube(data.groundtruth3DBB(kk),'r')
end

%%
anno2d = SUNRGBD2Dseg(imageId);
figure,
imagesc(anno2d.seglabel);
% category name in 37 categories list
load('./Metadata/seg37list.mat');
objectname37 = unique(anno2d.seglabel(:));
objectname37 = seg37list(objectname37(objectname37~=0));

figure,
imagesc(anno2d.seglabelall);
% category name of all categories
objectnameall = anno2d.names


%% Example to read single data
data = readframeSUNRGBD('/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/','/n/fs/sun3d/data/');
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
%% draw 
figure,
imshow(data.rgbpath);
hold on; 
for kk =1:length(data.groundtruth3DBB)
    rectangle('Position', [data.groundtruth3DBB(kk).gtBb2D(1) data.groundtruth3DBB(kk).gtBb2D(2) data.groundtruth3DBB(kk).gtBb2D(3) data.groundtruth3DBB(kk).gtBb2D(4)],'edgecolor','y');
    text(data.groundtruth3DBB(kk).gtBb2D(1),data.groundtruth3DBB(kk).gtBb2D(2),data.groundtruth3DBB(kk).classname,'BackgroundColor','y')
end
%% draw 3D 
figure,
vis_point_cloud(points3d,rgb)
hold on;
for kk =1:length(data.groundtruth3DBB)
   vis_cube(data.groundtruth3DBB(kk),'r')
end
