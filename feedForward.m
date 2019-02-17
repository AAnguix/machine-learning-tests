% Algorithm that calculates the output of a neural network of two layers.
function a3 = feedForward(theta1, theta2, X)

trainingExamples = size(X, 1);

% Add the bias term(a_0_1) to the X data matrix 
a1 = [ones(trainingExamples, 1) X];

% Compute the hidden layer
z2 = z2(theta1, a1);
a2 = sigmoid(z2);

% Add the bias term(a_0_2) to the a2 data matrix 
a2Cols = size(a2, 1);
a2 = [ones(a2Cols, 1) a2];

% Compute the output layer
z3 = z3(theta2, a2);
a3 = sigmoid(z3);

end

function z2 = z2(theta1, a1)
  z2 = computeLayer(theta1, a1);
end

function z3 = z3(theta2, a2)
  z3 = computeLayer(theta2, a2);
end

function output = computeLayer(weights, input)
  output = input*weights'; 
end