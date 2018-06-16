function [points2d,z3] = project3dPtsTo2d(points3d,Rtilt,crop,K)
    %% inverse of get_aligned_point_cloud
    points3d =[Rtilt'*points3d']';
    
    %% inverse rgb_plane2rgb_world
    
    
    % Now, swap Y and Z.
    points3d(:, [2, 3]) = points3d(:,[3, 2]);
    
    % Make the original consistent with the camera location:
    x3 = points3d(:,1);
    y3 = -points3d(:,2);
    z3 = points3d(:,3);
    
    xx = x3 * K(1,1) ./ z3 + K(1,3);
    yy = y3 * K(2,2) ./ z3 + K(2,3);
    

    xx = xx - crop(2) + 1;
    yy = yy - crop(1) + 1;

    points2d = [xx yy];
end