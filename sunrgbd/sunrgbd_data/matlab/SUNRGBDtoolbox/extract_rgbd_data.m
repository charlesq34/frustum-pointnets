%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes, point cloud (in camera
% coordinate), calibration and 3d boxes
%
% Author: Charles R. Qi
% Date: 09/27/2017
%
clear; close all; clc;
addpath(genpath('.'))
% load('./Metadata/SUNRGBDMeta.mat')
% load('./Metadata/SUNRGBD2Dseg.mat')
%load('../SUNRGBDMeta3DBB_v2.mat'); % SUNRGBDMeta2DBB
%load('../SUNRGBDMeta2DBB_v2.mat'); % SUNRGBDMeta
load('./Metadata/SUNRGBDMeta.mat'); % SUNRGBDMeta
depth_folder = 'mysunrgbd/depth/';
image_folder = 'mysunrgbd/image/';
calib_folder = 'mysunrgbd/calib/';
%label_folder = 'mysunrgbd/label/';
label_folder = 'mysunrgbd/label_dimension/';
mkdir(depth_folder);
mkdir(image_folder);
mkdir(calib_folder);
mkdir(label_folder);
%% Read
parfor imageId = 1:10335
    imageId
%     if imageId == 10
%         break
%     end
try
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('/data/rqi/SUNRGBD',data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('/data/rqi/SUNRGBD',data.rgbpath);

% Write point cloud in depth map
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
rgb(isnan(points3d(:,1)),:) = [];
points3d(isnan(points3d(:,1)),:) = [];
points3d_rgb = [points3d, rgb];
filename = strcat(num2str(imageId,'%06d'), '.txt');
dlmwrite(strcat(depth_folder, filename), points3d_rgb, 'delimiter', ' ');

% Write images
copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));

% Write calibration
dlmwrite(strcat(calib_folder, filename), data.Rtilt(:)', 'delimiter', ' ');
dlmwrite(strcat(calib_folder, filename), data.K(:)', 'delimiter', ' ', '-append');

% Write 2D and 3D box label
%data2d = SUNRGBDMeta2DBB(imageId);
data2d = data;
fid = fopen(strcat(label_folder, filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    %if data2d.groundtruth2DBB(j).has3dbox == 0
    %    continue
    %end
    centroid = data.groundtruth3DBB(j).centroid;
    classname = data.groundtruth3DBB(j).classname;
    orientation = data.groundtruth3DBB(j).orientation;
    coeffs = abs(data.groundtruth3DBB(j).coeffs);
    [new_basis, new_coeffs] = order_basis(data.groundtruth3DBB(j).basis, coeffs, centroid);
    box2d = data2d.groundtruth2DBB(j).gtBb2D;
    %assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), new_basis(1,1), new_basis(1,2), new_basis(2,1), new_basis(2,2), orientation(1), orientation(2));
end
fclose(fid);
catch
end

end
