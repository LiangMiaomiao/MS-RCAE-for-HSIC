function sae_features=stackedAE_fusionfeatures(train_data,fusion_data,block,param)
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
options.display = 'off';

[sae1_OptTheta, ~] = minFunc( @(p) GsparseAutoencoderCost(p, ...
                                   inputSize, param.hiddenSizeL1, ...
                                   param.lambda(1), param.sparsityParam(1), ...
                                   param.beta, train_data,block,param.alpha), ...
                                   sae1_Theta, options);
                               
sae1_Features = feedForwardAutoencoder(sae1_OptTheta, param.hiddenSizeL1, ...
                                        inputSize, train_data);
sae1_Features=data_preprocessing_nonzero(sae1_Features);
sae1_Features_=sae1_Features(:,1:end);
%% Train the 2 sparse autoencoder
param.hiddenSizeL1s=param.hiddenSizeL1+size(fusion_data,1);
sae2_Theta = initializeParameters(param.hiddenSizeL2, param.hiddenSizeL1s);
options.Method = 'lbfgs'; 
options.maxIter = 200;	   
options.display = 'off';

[sae2_OptTheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   param.hiddenSizeL1s, param.hiddenSizeL2, ...
                                   param.lambda(2), param.sparsityParam(2), ...
                                   param.beta, [sae1_Features_;fusion_data]), ...
                                   sae2_Theta, options);

sae2_Features = feedForwardAutoencoder(sae2_OptTheta, param.hiddenSizeL2, ...
                                        param.hiddenSizeL1s, [sae1_Features;fusion_data]);
sae_features=sae2_Features;