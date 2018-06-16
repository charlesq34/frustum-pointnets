mask = 5*double(instances(:,:,1))+5*double(labels(:,:,1));
maskColor = zeros(size(image,1)*size(image,2),3);
uniquemask = unique(mask(:));
for i =1:length(uniquemask)

    sel = mask(:)==uniquemask(i);

    maskColor(sel, :) = repmat(ObjectColor(uniquemask(i)),sum(sel),1);

end
maskColor(find(mask(:)==0),:) = 1;
maskColor = reshape(maskColor,[size(mask,1),size(mask,2),3]);


figure

imshow(maskColor);
imwrit(maskColor,'NYUmask.png')
%%
load('/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat')

addpath('/n/fs/modelnet/SUN3DV2/roomlayout/')
fullname = '/n/fs/sun3d/data/rgbd_voc/000414_2014-06-04_19-49-13_260595134347_rgbf000044-resize'
data = readframe(fullname);
groundTruthBbs  = data.groundtruth3DBB;

sequenceName = getSequenceName(fullname);
gtRoom3D = GroundTruthBox(sequenceName,0);
cameraXYZ = data.anno_extrinsics'*gtRoom3D;
cameraXYZ([2 3],:) = cameraXYZ([3 2],:);
cameraXYZ(3,:) = - cameraXYZ(3,:);
cameraXYZ = data.Rtilt * cameraXYZ;

my_mhCorner3D = cameraXYZ; %data.Rtilt*data.anno_extrinsics'*gtCorner3D;
visulize_wholeroom(groundTruthBbs,cls,fullname,my_mhCorner3D)
  