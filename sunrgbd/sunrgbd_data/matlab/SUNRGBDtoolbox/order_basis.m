function [new_basis, new_coeffs] = order_basis(basis, coeffs, centroid)
  % Order the bases.
  [~, inds] = sort(abs(basis(:,1)), 'descend');
  basis = basis(inds, :);
  coeffs = coeffs(inds);
  
  [~, inds] = sort(abs(basis(2:3,2)), 'descend');
  if inds(1) == 2
    basis(2:3,:) = flipdim(basis(2:3,:), 1);
    coeffs(2:3) = flipdim(coeffs(2:3), 2);
  end
  
  % Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis
  % vectors towards the viewer.
  new_basis = flip_towards_viewer(basis, repmat(centroid, [3 1]));
  new_coeffs = coeffs;
end

function normals = flip_towards_viewer(normals, points)
  points = points ./ repmat(sqrt(sum(points.^2, 2)), [1, 3]);
  
  proj = sum(points .* normals, 2);
  
  flip = proj > 0;
  normals(flip, :) = -normals(flip, :);
end
