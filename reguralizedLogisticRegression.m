% Computes cost and gradient for logistic regression with regularization.
function [J, grad] = reguralizedLogisticRegression(theta, X, y, lambda)

m = length(y); % Number of training examples

J = 0;
grad = zeros(size(theta));

z = X*theta;
h = sigmoid(z);
% Each row of the resulting matrix h, will contain the value of the
% prediction for that example.

first = -y'*(log(h));
second = (1-y')*log(1-h);
total = first - second;
computed = sum(total);

%Calculate regularized term
thetaOneToEnd = theta(2:end); %theta(1) is theta cero, and should not be regularized
sumOfSquaredThetas = sum(thetaOneToEnd.^2);
regThetas = lambda/(2*m)*sumOfSquaredThetas;

%Cost
J = ((1/m) * computed) + regThetas;

%Gradient
unregularizedGrad = ((X')*(h-y))/ m; %Unregularized gradient for logistic regression
thetaWithoutBias = theta; 
thetaWithoutBias(1) = 0;   % Because we don't add anything for j = 0

gradRegTerm = (lambda/m).*thetaWithoutBias;
grad = unregularizedGrad + gradRegTerm;
grad = grad(:);

end