function [cost,grad] = GsparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data,block,alpha)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: dim*n matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 
%% initialization parameter
% W1 is a hiddenSize * visibleSize matrix
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 is a visibleSize * hiddenSize matrix
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 is a hiddenSize * 1 vector
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 is a visible * 1 vector
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
%%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                   and the corresponding gradients W1grad, W2grad, b1grad, b2grad,which should be computed using backpropagation.
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
numCases = size(data, 2);
%% forward propagation
y_l = W1 * data + repmat(b1, 1, numCases); 
y_s = sigmoid(y_l); %%
z_l = W2 * y_s + repmat(b2, 1, numCases);
z_s = sigmoid(z_l);%%

% error
sqrerror = (data - z_s) .* (data - z_s); 
error = sum(sum(sqrerror)) / (2 * numCases);
% weight decay
wtdecay = (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2))) / 2;
% sparsity
rho = sum(y_s, 2) ./ numCases;
divergence = sparsityParam .* log(sparsityParam ./ (rho+eps)) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - rho+eps));
sparsity = sum(divergence);
% G constraint
G_err=0;
for i=1:numel(block)
    if length(block{i})>2
        H_=repmat(mean(y_s(:,block{i}),2),1,length(block{i}));
        G_err=G_err+1/length(block{i})*(sum(sum(y_s(:,block{i})-H_)));
    end
end
G_err=1/numel(block)*G_err;
% objective function
cost = error + lambda * wtdecay + beta * sparsity + alpha * G_err;
%% derivative
% delta3 is a visibleSize * numCases matrix
delta3 = -(data - z_s).* z_s.*(1-z_s); %%
% delta2 is a hiddenSize * numCases matrix
sparsityterm = beta * (-sparsityParam ./ (rho+eps) + (1-sparsityParam) ./ (1-rho+eps));
delta2 = (W2' * delta3 + repmat(sparsityterm, 1, numCases)) .* y_s.*(1-y_s); 

G_Wdelta=0; G_bdelta=0; block_num=0;
for i=1:numel(block)
    if length(block{i})>2
        H_=repmat(mean(y_s(:,block{i}),2),1,length(block{i}));
        X_=repmat(mean(data(:,block{i}),2),1,length(block{i}));
    G_Wdelta=G_Wdelta+1/(2*length(block{i}))*(((y_s(:,block{i})-H_).*(y_s(:,block{i}).*(1-y_s(:,block{i}))))*data(:,block{i})'...
        -((y_s(:,block{i})-H_).*(H_.*(1-H_)))*X_');
    Gb_=1/(2*length(block{i}))*(y_s(:,block{i})-H_).*(y_s(:,block{i}).*(1-y_s(:,block{i}))-H_.*(1-H_));
    G_bdelta=G_bdelta+sum(Gb_, 2);
    block_num=block_num+1;
    end
end
G_Wdelta=1/block_num*G_Wdelta;
G_bdelta=1/block_num*G_bdelta;

W1grad = delta2 * data' ./ numCases + lambda * W1 + alpha * G_Wdelta; 
b1grad = sum(delta2, 2) ./ numCases + alpha * G_bdelta;

W2grad = delta3 * y_s' ./ numCases + lambda * W2; 
b2grad = sum(delta3, 2) ./ numCases; 

% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
