% Returns the partial derivatives of the cost function with respect to theta1 and theta2.
% This code works for a two layer neural network.
function [thetha1_grad, thetha2_grad] = backpropagation(X, y, theta1, theta2, numOflabels)
  
thetha1_grad = zeros(size(theta1));
thetha2_grad = zeros(size(theta2));

trainingExamples = size(X, 1);

Delta1 = 0;
Delta2 = 0;

for te=1:trainingExamples
  a1 = X(te,:); 
  
  % Add the bias term
  a1 = [1,a1];

  % Compute the hidden layer
  z2 = computeLayer(theta1, a1);
  a2 = sigmoid(z2);

  % Add the bias term
  a2 = [1,a2];

  % Compute the output layer
  z3 = computeLayer(theta2, a2);
  a3 = sigmoid(z3);
  
  expectedY = y(te,:);
  y_k = (binarize(expectedY, numOflabels))';
  
  delta3 = a3 - y_k;
  delta2 = hiddenUnitDelta(delta3, theta2, z2);

  Delta1 = Delta1 + (delta2'*a1);
  Delta2 = Delta2 + (delta3'*a2);
end

thetha1_grad = (1/trainingExamples)*Delta1;
thetha2_grad = (1/trainingExamples)*Delta2;

end

% Computes the error of a neural network hidden layer (l).
% nextLayerDelta: Weighted average of the error terms of the nodes in layer l+1
% currentTheta: Weights of the hidden layer (l).
function delta = hiddenUnitDelta(nextLayerDelta, currentTheta, z)
  z = [1,z];
  delta = (nextLayerDelta*currentTheta).*sigmoidGradient(z);
  %Skip delta2_0
  delta = delta(2:end);
end