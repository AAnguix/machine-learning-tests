function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix (a_0_1)
a1 = [ones(m, 1) X];

%Compute the hidden layer
z2 = z2(Theta1, a1);
a2 = sigmoid(z2);

% Add ones to the a2 data matrix (a_0_2)
a2Cols = size(a2, 1);
a2 = [ones(a2Cols, 1) a2];

%Compute the output layer
z3 = z3(Theta2, a2);
a3 = sigmoid(z3);

% Pick the max value of the 10 columns for each row.
% The column indicates the label (predicted value). 
[maxPrediction, maxIndex] = max(a3, [], 2);
p = maxIndex;
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