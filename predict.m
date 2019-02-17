% Returns the predicted label of an input given the parameters/weights of a trained neural network.
% theta1: Weights of the hidden layer.
% theta2: Weights of the output layer.
function predictedLabel = predict(theta1, theta2, X)

a3 = feedForward(theta1, theta2, X);
% Pick the maximum value of all the columns for each row.
% The column indicates the label (predicted value). 
[maxPrediction, maxIndex] = max(a3, [], 2);
predictedLabel = maxIndex;

end
