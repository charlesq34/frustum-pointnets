% Returns the average precision score given precision and recall vectors. The AP is computed by
% numerically integrating the area under the PR curve.
%
% Note: this code was taken from the VOC2011 toolkit.
%
% Args:
%   precision - Px1 vector of precision scores, where P is the number of predictions. Note that the
%               precision scores must be monotonically decreasing.
%   recall - Px1 vector of recall scores, where P is the number of predictions.
%
% Returns:
%   ap - the average precision.
function ap = get_average_precision(precision, recall)

  mrec = [0; recall; 1];
  mpre = [0; precision; 0];
  
  for ii = numel(mpre) - 1 : -1 : 1
    mpre(ii) = max(mpre(ii), mpre(ii+1));
  end
  
  ii = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
  ap = sum((mrec(ii) - mrec(ii-1)) .* mpre(ii));
end