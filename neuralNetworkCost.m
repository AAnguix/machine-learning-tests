%Calculates the cost of a neural network
function J = neuralNetworkCost(X, y, theta1, theta2, numOflabels)

% Setup some useful variables
trainingExamples = size(X, 1);
totalCost = 0;

for i = 1:trainingExamples
  x = X(i,:);
  expectedY = y(i,:);
  binarizedY = binarize(expectedY, numOflabels);
  j = cost(x, binarizedY, theta1, theta2);
  totalCost = totalCost + j; 
end

J = 1/trainingExamples * totalCost;

end

%Returns the cost of and specific example
function J = cost(x, y, theta1, thetha2)
  h = feedForward(theta1, thetha2, x);

  first = log(h)*-y;
  second = log(1-h)*(1-y);

  total = first - second;
  J = sum(total);
end