% Helper method for quickly creating bounding boxes.
%
% Args:
%   basis2d - 2x2 matrix for the basis in the XY plane
%   centroid - 1x3 vector for the 3D centroid of the bounding box.
%   coeffs - 1x3 vector for the radii in each dimension (x, y, and z)
%
% Returns:
%   bb - a bounding box struct.
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function bb = create_bounding_box_3d(basis2d, centroid, coeffs)
  assert(all(size(basis2d) == [2, 2]));
  assert(numel(centroid) == 3);
  assert(numel(coeffs) == 3);
  
  centroid = centroid(:)';
  coeffs = coeffs(:)';

  bb = struct();
  bb.basis = zeros(3,3);
  bb.basis(3,:) = [0 0 1];
  bb.basis(1:2,1:2) = basis2d;
  
  bb.centroid = centroid;
  bb.coeffs = coeffs;
%   bb.volume = prod(2 * bb.coeffs);
end