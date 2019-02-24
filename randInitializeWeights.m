%Returns the randomly initialized weights of a layer
%incomingConnections: number of incoming connections in the layer.
%outgoingConnections: number of outgoing connections in the layer.
%W: matrix of size(outgoing, 1 + incoming), as the first column of W handles the "bias" terms.
% This is used to break the symmetry while training a neural network.
function W = randInitializeWeights(incomingConnections, outgoingConnections)

initEpsilon = 0.12;
W = rand(outgoingConnections, 1 + incomingConnections) * 2 * initEpsilon - initEpsilon;

end
