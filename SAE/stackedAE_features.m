function sae_features=stackedAE_features(train_data,block,param)
% param.hiddenSizeL1;     Layer 1 Hidden Size
% param.hiddenSizeL2;     Layer 2 Hidden Size
% param.lambda;           weight decay parameter       
% param.beta;             weight of sparsity penalty term 
% param.sparsityParam ;   desired average activation of the hidden units.
                          % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		                  %  in the lecture notes). 
%% Train the 1 sparse autoencoder
inputSize = size(train_data,1);
sae1_Theta = initializeParameters(param.hiddenSizeL1, inputSize);
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'off'; % on
[sae1_OptTheta, ~] = minFunc( @(p) GsparseAutoencoderCost(p, ...
                                   inputSize, param.hiddenSizeL1, ...
                                   param.lambda, param.sparsityParam, ...
                                   param.beta, train_data,block,param.alpha), ...
                              sae1_Theta, options);
[sae1_Features] = feedForwardAutoencoder(sae1_OptTheta, param.hiddenSizeL1, ...
                                        inputSize, train_data);
sae_features=sae1_Features;