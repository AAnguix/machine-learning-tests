%Returns an array of 0's of 'labels' length, with 1 in the 'y' index.
% Example (y=3, labels = 4) : [0, 0, 1, 0]
function binarizedY = binarize(y, labels)
  binarizedY = zeros(labels, 1);
  binarizedY(y) = 1;
end