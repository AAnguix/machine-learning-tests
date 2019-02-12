%alpha: The learning rate.
%iterations: The number of times that the algorith will be executed.
% Returns the learned theta by taking iterations of gradient steps with a learning rate.
function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)

trainingExamples = length(y);
J_history = zeros(iterations, 1);

for iter = 1:iterations
  
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    x = X(:,2);
    h = theta(1) + (theta(2)*x);
    
    tempTheta0 = theta(1) - alpha * (1/trainingExamples) * sum(h-y);
    tempTheta1 = theta(2) - alpha * (1/trainingExamples) * sum((h - y) .* x);
     
    theta(1) = tempTheta0;
    theta(2) = tempTheta1;
end
end