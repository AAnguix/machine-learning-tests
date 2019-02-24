% Returns the gradient of the sigmoid function evaluated at z. 
% z: Scalar, vector or matrix (in the last two, it returs the gradient for each element).
function g = sigmoidGradient(z)

sig = sigmoid(z);
g = sig.*(1.-sig);

end