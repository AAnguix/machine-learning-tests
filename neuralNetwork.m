% Returns the cost and gradient for a two layer neural network.
% neural network which performs classification
% -Parameters-
% unrolledWeights: vector of "unrolled" parameters of the neural network. 
% They need to be converted back into the weight matrices.
% y: vector of labels containing values from 1..K.
% It needs to be mapped into a binary vector of 1's and 0's to be used with the neural network cost function.
% -Returns-
% J: The neural network regularized cost.
% grad: the "unrolled" vector of the partial derivatives of the neural network.
function [J grad] = neuralNetwork(unrolledWeights, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Reshape unrolledWeights back into the parameters theta1 and theta2, the weight matrices
% for our 2 layer neural network
theta1 = reshape(unrolledWeights(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(unrolledWeights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

J = neuralNetworkCost(X, y, theta1, theta2, num_labels);
trainingExamples = size(X, 1);
regularizedJ = neuralNetworkRegularizedCost(trainingExamples, theta1, theta2, lambda, 
input_layer_size, hidden_layer_size, num_labels);

J = J + regularizedJ;

[theta1_grad, theta2_grad] = backpropagation(X, y, theta1, theta2, num_labels);

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];
end