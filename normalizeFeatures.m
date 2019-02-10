% Returns a normalized version of X where 
% the mean value of each feature is 0 and the standard deviation is 1.
% X: Matrix where each column is a feature and each row is an example.
function [normalizedX, mu, sigma] = normalizeFeatures(X)

columns = size(X,2);
normalizedX = X;
for col = 1:columns
  x = X(:,col);
  mu = mean(x);
  sigma = std(x);
  normalizedX(:,col) = (x.-mu)./sigma;
end

end