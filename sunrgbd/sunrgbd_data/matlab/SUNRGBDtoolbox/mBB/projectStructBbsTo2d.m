function [bb2d,bb2dDraw] = projectStructBbsTo2d(bb,Rtilt,crop,K)
    if isempty(crop)
       crop =[1,1];
    end
    if isempty(bb),
        bb2d =[];
        bb2dDraw =[];
    else
        nBbs = numel(bb);
        if isfield(bb,'confidence'),
            conf = [bb.confidence];
            conf = conf(:);
        else
            conf = ones(nBbs,1);
        end
        points3d = zeros(8*nBbs,3);
        for i = 1:nBbs,
            corners = get_corners_of_bb3d(bb(i));
            points3d((i-1)*8+(1:8),:) = corners([8 4 5 1 7 3 6 2],:);
            %points3d((i-1)*8+(1:8),:) = corners([5,1,8,4,6,2,3,7],:);
        end
        
        points2d = project3dPtsTo2d(points3d,Rtilt,crop,K);

        bb2d = zeros(nBbs,5);
        bb2d(:,1) = min(reshape(points2d(:,1),[8,nBbs]),[],1);
        bb2d(:,2) = min(reshape(points2d(:,2),[8,nBbs]),[],1);
        bb2d(:,3) = max(reshape(points2d(:,1),[8,nBbs]),[],1);
        bb2d(:,4) = max(reshape(points2d(:,2),[8,nBbs]),[],1);
        bb2d(:,3) = bb2d(:,3) - bb2d(:,1);
        bb2d(:,4) = bb2d(:,4) - bb2d(:,2);
        bb2d(:,5) = conf;

        bb2dDraw = zeros(nBbs,17);
        pts = points2d';
        pts = pts(:);
        pts = reshape(pts,[16,nBbs]);
        bb2dDraw(:,1:16) = pts';
        bb2dDraw(:,17) = conf;
    end
end