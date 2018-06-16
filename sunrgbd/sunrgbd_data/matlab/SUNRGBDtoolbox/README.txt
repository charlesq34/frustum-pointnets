****************************************************************************************
Data: Image depth and label data are in SUNRGBD.zip
image: rgb image
depth: depth image  to read the depth see the code in SUNRGBDtoolbox/read3dPoints/.
extrinsics: the rotation matrix to align the point could with gravity
fullres: full resolution depth and rgb image
intrinsics.txt  : sensor intrinsic
scene.txt  : scene type
annotation2Dfinal  : 2D segmentation
annotation3Dfinal  : 3D bounding box
annotation3Dlayout : 3D room layout bounding box

****************************************************************************************
Label: 
In SUNRGBDtoolbox/Metadata 
SUNRGBDMeta.mat:  2D,3D bounding box ground truth and image information for each frame.
SUNRGBD2Dseg.mat:  2D segmetation ground truth. 
The index in "SUNRGBD2Dseg(imageId).seglabelall"  mapping the name to "seglistall". 
The index in "SUNRGBD2Dseg(imageId).seglabel" are mapping the object name in "seg37list".
 
****************************************************************************************
In SUNRGBDtoolbox/traintestsplit
allsplit.mat: stores the training and testing split.

****************************************************************************************
Code:
SUNRGBDtoolbox/demo.m : Examples to load and visualize the data.
SUNRGBDtoolbox/readframeSUNRGBD.m : Example code to read SUNRGBD annotation from ".json" file.

*****************************************************************************************
Citation:
Please cite our paper if you use this data:
S. Song, S. Lichtenberg, and J. Xiao.
SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite
Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)


