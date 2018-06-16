function visulize_wholeroom(BbsTight,cls,fullname,roomLayout,numofpoints2plot,savepath)
        if ~exist('numofpoints2plot','var'),
            numofpoints2plot = 5000;
        end
        sizeofpoint = max(round(200000/numofpoints2plot),3);
        Linewidth =5;
        maxhight = 1.2;
        vis = 'on'
        f = figure;
        set(f, 'Position', [100, 100, 1149, 1249]);
        if exist('fullname','var')&&~isempty(fullname);
            data = readframe(fullname);
            [rgb,points3d,~,imsize]=read3dPoints(data);
            vis_point_cloud(points3d,double(rgb),sizeofpoint,numofpoints2plot);
            maxhight = min(maxhight,max(points3d(:,3)));
            hold on;
        end
        if ~isempty(BbsTight)
            if exist('cls','var')&&~isempty(cls)&&~isfield(BbsTight,'classid')
                [~,classid] = ismember({BbsTight.classname},cls);
            else
                classid = [BbsTight.classid];
            end
        end
        for i =1:length(BbsTight)
            vis_cube(BbsTight(i), myObjectColor(classid(i)),Linewidth);
        end
        hold on;
        if exist('roomLayout','var')&&~isempty(roomLayout)
           drawRoom(roomLayout,'b',Linewidth,maxhight);
        end
        axis equal;
        axis tight;
        axis off;
        view(16,32);
        if exist('savepath','var')&&~isempty(savepath)
           saveas(f,savepath);
           close(f)
        end

end