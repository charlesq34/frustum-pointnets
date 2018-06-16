% Visualizes a 3D bounding box.
%
% Args:
%   bb3d - 3D bounding box struct
%   color - matlab color code, a single character
%   lineWidth - the width of each line of the square
%
% See:
%   create_bounding_box_3d.m
%
% Author:
%   Nathan Silberman (silberman@cs.nyu.edu)
function vis_cube(bb3d, color, lineWidth)
  if nargin < 3
    lineWidth = 0.5;
  end
  corners = get_corners_of_bb3d(bb3d);
  draw_square_3d(corners, color, lineWidth);
end