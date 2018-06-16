% Visualizes a line in 2D or 3D space
% 
% Args:
%   p1 - 1x2 or 1x3 point
%   p2 - 1x2 or 1x3 point
%   color - matlab color code, a single character
%   lineWidth - the width of the drawn line
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function vis_line(p1, p2, color, lineWidth)
  if nargin < 3
    color = 'b';
  end
  
  if nargin < 4
    lineWidth = 0.5;
  end
  
  % Make sure theyre the same size.
  assert(ndims(p1) == ndims(p2), 'Vectors are of different dimensions');
  assert(all(size(p1) == size(p2)), 'Vectors are of different dimensions');
  
  switch numel(p1)
    case 2
      line([p1(1) p2(1)], [p1(2) p2(2)], 'Color', color);
    case 3
      line([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'Color', color, 'LineWidth', lineWidth);
    otherwise
      error('vectors must be either 2 or 3 dimensional');
  end
end
