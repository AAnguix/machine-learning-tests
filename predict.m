function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add ones to the X data matrix (a_0_1)
a1 = [ones(m, 1) X];
z2 = z2(Theta1, a1);

%Compute the hidden layer
a2 = a2(z2);

% Add ones to the a2 data matrix (a_0_2)
a2Cols = size(a2, 1);
a2 = [ones(a2Cols, 1) a2];

z3 = z3(Theta2, a2);

%Compute the output layer
a3 = a3(z3);

% Pick the max value of the 10 columns for each row.
% The column indicates the label (predicted value). 
[maxPrediction, maxIndex] = max(a3, [], 2);
p = maxIndex;
% =========================================================================
end

function z2 = z2(theta1, a1)
  z2 = a1*theta1';
end

function a2 = a2(z2)
  a2 = sigmoid(z2);
end

function z3 = z3(theta2, a2)
  z3 = a2*theta2';
end

function a3 = a3(z3)
  a3 = sigmoid(z3);
end
