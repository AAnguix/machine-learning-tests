% Computes the linear regression cost(J) to fit the data points in X and y,
% using theta as the parameter.
function J = linearRegression(X, y, theta)

trainingExamples = length(y); 

predictions = X*theta; %Predictions of hypothesis on all training examples
squareErrors = (predictions-y).^2;
squareErrorsSum = sum(squareErrors);
temp = 1/(2*trainingExamples);

J = temp * squareErrorsSum;

end
