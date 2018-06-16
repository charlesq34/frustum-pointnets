function data = readframeSUNRGBD(thispath,dataRoot,cls,bbmode)  
    % example code to read annotation from ".json" file.
    % thispath: full path to the data folder.
    % dataRoot: root directory of all data folder.
    % cls : object category of ground truth to load. If not speficfy, the code will load all ground truth. 
    if ~exist('cls','var')
        cls  =[];
    end
    if ~exist('bbmode','var')
        bbmode  ='2Dbb';
    end

    if ~exist('dataRoot','var')||isempty(dataRoot)
        dataRoot = '/n/fs/sun3d/data/';
    end
    sequenceName  = getSequenceName(thispath,dataRoot);
    if ~exist(thispath,'dir')
        data.sequenceName = sequenceName;
        data.valid = 0;
        return;
    end
    indd = find(sequenceName=='/');
    sensorType = sequenceName(indd(1)+1:indd(2)-1);
    % get K
    fID = fopen([thispath '/intrinsics.txt'],'r');
    K = reshape(fscanf(fID,'%f'),[3,3])';
    fclose(fID);
    
     % get image and depth path
    depthpath = dir([thispath '/depth/' '/*.png']);
    depthname = depthpath(1).name;
    depthpath = [thispath '/depth/' depthpath(1).name];
    
    rgbpath = dir([thispath '/image/' '/*.jpg']);
    rgbname = rgbpath(1).name;
    rgbpath = [thispath '/image/' rgbpath(1).name];
     
    if exist(sprintf('%s/annotation3Dfinal/index.json',thispath),'file')
        annoteImage =loadjson(sprintf('%s/annotation3Dfinal/index.json',thispath));
        % get Box
        filename = dir([fullfile(thispath,'extrinsics') '/*.txt']);
        Rtilt = dlmread([fullfile(thispath,'extrinsics') '/' filename(end).name]);
        Rtilt = Rtilt(1:3,1:3);
        anno_extrinsics = Rtilt;
        % convert it into matlab coordinate
        Rtilt = [1 0 0; 0 0 1 ;0 -1 0]*Rtilt*[1 0 0; 0 0 -1 ;0 1 0];


        cnt =1;
        for obji =1:length(annoteImage.objects)
                annoteobject =annoteImage.objects(obji);
                if ~isempty(annoteobject)&&~isempty(annoteobject{1})&&~isempty(annoteobject{1}.polygon)
                    annoteobject =annoteobject{1};
                    box = annoteobject.polygon{1};

                    % class name and label 
                    ind = find(annoteobject.name==':');
                    if isempty(ind)
                        classname  = annoteobject.name;
                        labelname ='';
                    else
                        if ismember(annoteobject.name(ind-1),{'_',' '}),
                            clname = annoteobject.name(1:ind-2);
                        else
                            clname = annoteobject.name(1:ind-1);
                        end
                        %[~,classId]= ismember(clname,classNames);
                        classname  = clname;
                        labelname = annoteobject.name(ind+2:end);
                        %[~,label]= ismember(Labelname,labelNames);
                    end
                    if ismember(classname,{'wall','floor','ceiling'})||(~isempty(cls)&&~(sum(ismember(cls,{classname}))>0)), 
                        continue;
                    end


                    x =box.X;
                    y =box.Z;
                    vector1 =[x(2)-x(1),y(2)-y(1),0];
                    coeff1 =norm(vector1);
                    vector1 =vector1/norm(vector1);
                    vector2 =[x(3)-x(2),y(3)-y(2),0];
                    coeff2 = norm(vector2);
                    vector2 =vector2/norm(vector2);
                    up = cross(vector1,vector2);
                    vector1 = vector1*up(3)/up(3);
                    vector2 = vector2*up(3)/up(3);
                    zmax =-box.Ymax;
                    zmin =-box.Ymin;
                    centroid2D = [0.5*(x(1)+x(3)); 0.5*(y(1)+y(3))];

                    thisbb.basis = [vector1;vector2; 0 0 1]; % one row is one basis
                    thisbb.coeffs = abs([coeff1, coeff2, zmax-zmin])/2;
                    thisbb.centroid = [centroid2D(1), centroid2D(2), 0.5*(zmin+zmax)];
                    thisbb.classname = classname;
                    thisbb.labelname = labelname;
                    thisbb.sequenceName = sequenceName;
                    orientation = [([0.5*(x(2)+x(1)),0.5*(y(2)+y(1))] - centroid2D(:)'), 0];
                    thisbb.orientation = orientation/norm(orientation);

                    if strcmp(bbmode,'2Dbb'),
                        [bb2d,bb2dDraw] = projectStructBbsTo2d(thisbb,Rtilt,[],K);
                        %gtBb2D = crop2DBB(gtBb2D,427,561);
                        thisbb.gtBb2D = bb2d(1:4);
                    end
                    groundtruth3DBB(cnt) =thisbb;
                    cnt=cnt+1;
                end
        end
        if cnt==1,groundtruth3DBB =[];end
    else
        groundtruth3DBB =[];
        filename = dir([fullfile(thispath,'extrinsics') '/*.txt']);
        Rtilt = dlmread([fullfile(thispath,'extrinsics') '/' filename(end).name]);
        Rtilt = Rtilt(1:3,1:3);
        anno_extrinsics = Rtilt;
        Rtilt = [1 0 0; 0 0 1 ;0 -1 0]*Rtilt*[1 0 0; 0 0 -1 ;0 1 0];
         
    end
    % read in room 
    gtCorner3D =[];
    if exist([thispath '/annotation3Dlayout/index.json'],'file')
       json=loadjson([thispath '/annotation3Dlayout/index.json']);
       for objectID=1:length(json.objects)
           try
                groundTruth = json.objects{objectID}.polygon{1};
                numCorners = length(groundTruth.X);

                gtCorner3D(1,:) = [groundTruth.X groundTruth.X];
                gtCorner3D(2,:) = [repmat(groundTruth.Ymin,[1 numCorners]) repmat(groundTruth.Ymax,[1 numCorners])];
                gtCorner3D(3,:) = [groundTruth.Z groundTruth.Z];
                gtCorner3D = anno_extrinsics'*gtCorner3D;
                gtCorner3D = gtCorner3D([1,3,2],:);
                gtCorner3D(3,:) = -1*gtCorner3D(3,:);
                gtCorner3D = Rtilt*gtCorner3D;
                break;
           catch
           end
       end
       
    end
     
    data =struct('sequenceName',sequenceName,'groundtruth3DBB',...
         groundtruth3DBB,'Rtilt',Rtilt,'K',K,...
         'depthpath',depthpath,'rgbpath',rgbpath,'anno_extrinsics',anno_extrinsics,'depthname',depthname,...
         'rgbname',rgbname,'sensorType',sensorType,'valid',1,'gtCorner3D',gtCorner3D);

end