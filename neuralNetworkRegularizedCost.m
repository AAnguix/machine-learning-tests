%Calculates the regularized cost of a neural network
function J = neuralNetworkRegularizedCost(trainingExamples, theta1, theta2, lambda, 
  inputLayerSize, hiddenLayerSize, labels)
  regTerm = lambda/(2*trainingExamples);
  theta1WithoutBias = theta1(:,2:end); %the first column is the bias term, and should not be regularized
  theta2WithoutBias = theta2(:,2:end);
  sqrTheta1 = theta1WithoutBias.^2;
  sqrTheta2 = theta2WithoutBias.^2;
  first = sum(sum(sqrTheta1));
  second = sum(sum(sqrTheta2));
  summaries = first + second;
  J = regTerm * summaries;
end