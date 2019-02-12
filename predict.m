% Returns the predicted label of an input given the parameters/weights of a trained neural network.
% theta1: Weights of the hidden layer.
% theta2: Weights of the output layer.
function predictedLabel = predict(theta1, theta2, X)

trainingExamples = size(X, 1);
num_labels = size(theta2, 1);

predictedLabel = zeros(size(X, 1), 1);

% Add the bias term(a_0_1) to the X data matrix 
a1 = [ones(trainingExamples, 1) X];

%Compute the hidden layer
z2 = z2(theta1, a1);
a2 = sigmoid(z2);

% Add the bias term(a_0_2) to the a2 data matrix 
a2Cols = size(a2, 1);
a2 = [ones(a2Cols, 1) a2];

%Compute the output layer
z3 = z3(theta2, a2);
a3 = sigmoid(z3);

% Pick the maximum value of all the columns for each row.
% The column indicates the label (predicted value). 
[maxPrediction, maxIndex] = max(a3, [], 2);
predictedLabel = maxIndex;
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