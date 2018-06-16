% Draws a square in 3D
%
% Args:
%   corners - 8x2 matrix of 2d corners.
%   color - matlab color code, a single character.
%   lineWidth - the width of each line of the square.
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function draw_square_3d(corners, color, lineWidth)
  if nargin < 2
    color = 'r';
  end
  
  if nargin < 3
    lineWidth = 0.5;
  end

  vis_line(corners(1,:), corners(2,:), color, lineWidth);
  vis_line(corners(2,:), corners(3,:), color, lineWidth);
  vis_line(corners(3,:), corners(4,:), color, lineWidth);
  vis_line(corners(4,:), corners(1,:), color, lineWidth);
  
  vis_line(corners(5,:), corners(6,:), color, lineWidth);
  vis_line(corners(6,:), corners(7,:), color, lineWidth);
  vis_line(corners(7,:), corners(8,:), color, lineWidth);
  vis_line(corners(8,:), corners(5,:), color, lineWidth);
  
  vis_line(corners(1,:), corners(5,:), color, lineWidth);
  vis_line(corners(2,:), corners(6,:), color, lineWidth);
  vis_line(corners(3,:), corners(7,:), color, lineWidth);
  vis_line(corners(4,:), corners(8,:), color, lineWidth);
end
