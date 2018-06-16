function scoreMatrix = bb3dOverlapCloseForm(bb1input,bb2struct)
% to run 
% bb1 =[0,0,0,1,1,1];load('bb3d'); bb3dOverlapCloseForm(bb1,bb3d);
% bb1 can be  nx6 bb or struct as bb2struct
% convert BB to format : x1 y1 x2 y2 x3 y3 x4 y4 zMin zMax    
if isempty(bb1input)||isempty(bb2struct)
    scoreMatrix =[];
    return;
end
    nBb1 = size(bb1input,1);
    nBb2 = size(bb2struct,1);
    if size(bb1input,2)>=6&&size(bb1input,2)<10
        bb1 = bb1input;
        bb1(:,4:6) = bb1input(:,4:6) + bb1input(:,1:3);    
        xMax = bb1(:,4);
        yMax = bb1(:,5);
        bb1 = [bb1(:,1) bb1(:,2) xMax bb1(:,2) xMax yMax bb1(:,1) yMax bb1(:,3) bb1(:,6)];
    elseif size(bb1input,2)==1
        for i = 1:nBb1
            corners = get_corners_of_bb3d(bb1input(i));
            bb1(i,:) = [reshape([corners(1:4,1) corners(1:4,2)]',1,[]) min(corners([1 end],3)) max(corners([1 end],3))];
        end
    elseif size(bb1input,2)>=10
        bb1 = bb1input(:,1:10);
    end
    
    for i = 1:nBb2
        corners = get_corners_of_bb3d(bb2struct(i));
        bb2(i,:) = [reshape([corners(1:4,1) corners(1:4,2)]',1,[]) min(corners([1 end],3)) max(corners([1 end],3))];
    end
    
    bb1 = bb1';
    bb2 = bb2';
    
    % a ha, we are done with dirty format conversion
    
    
    nBb1 = size(bb1,2);
    nBb2 = size(bb2,2);
    
    volume1 = cuboidVolume(bb1);
    volume2 = cuboidVolume(bb2);
    intersection = cuboidIntersectionVolume(double(bb1),double(bb2));
    
    %{
    volume1(6818)
    volume2(1)
    intersection(6818,1)
    cuboidVolume(bb1(:,6818))
    cuboidVolume(bb2(:,1))
    cuboidIntersectionVolume(bb1(:,6818),bb2(:,1))
    cuboidDraw(bb1(:,6818))
    cuboidDraw(bb2(:,1))
    %}
    
    union = repmat(volume1',1,nBb2)+repmat(volume2,nBb1,1)-intersection;
    
    scoreMatrix = intersection ./ union;
end