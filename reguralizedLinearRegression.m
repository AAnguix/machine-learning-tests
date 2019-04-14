% Computes cost and gradient for linear regression with regularization.
function [J, grad] = reguralizedLinearRegression(X, y, theta, lambda)

trainingExamples = length(y);

firstTerm = linearRegression(X, y, theta);

%Calculate regularized term
thetaOneToEnd = theta(2:end); %theta(1) is theta cero, and should not be regularized
sumOfSquaredThetas = sum(thetaOneToEnd.^2);
regThetas = (lambda/(2*trainingExamples))*sumOfSquaredThetas;

%Cost
J = firstTerm + regThetas;

%Gradient
h = X*theta;
unregularizedGrad = ((X')*(h-y))/trainingExamples;
thetaWithoutBias = theta; 
thetaWithoutBias(1) = 0;   % Because we don't add anything for j = 0

gradRegTerm = (lambda/trainingExamples).*thetaWithoutBias;
grad = unregularizedGrad + gradRegTerm;
grad = grad(:);

end
