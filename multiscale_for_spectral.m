run matconvnet/matlab/vl_setupnn ;
addpath matconvnet
addpath data;
%% Data pretreat
datasets = 'paviaU';% paviaU; salinas; indian_pines; KSC
inputVar = 'input' ;

HSI=double(cell2mat(struct2cell(load(sprintf('data/HSI/%s.mat',datasets)))));

HSI=data_preprocessing(HSI);

% sz = [size(HSI,1), size(HSI,2)] ;
% sz_ = round(sz / 32)*32 ; 
% HSI = imresize(HSI, sz_,'nearest') ; Worse Outcomes

if strcmp(datasets,'paviaU')
    HSI=HSI(1:end-2,:,:);
    HSI=[HSI(:,1:6,:),HSI,HSI(:,end-5:end,:)];
elseif strcmp(datasets,'indian_pines')
    HSI=[HSI(1:7,:,:);HSI;HSI(end-7:end,:,:);];
    HSI=[HSI(:,1:7,:),HSI,HSI(:,end-7:end,:)];
elseif strcmp(datasets,'salinas')
    HSI=[HSI(:,1:3,:),HSI,HSI(:,end-3:end,:)];
end
%% pooling net
net.layers = {} ;
net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [2 2], 'stride', 2, 'pad', 0);
net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [2 2], 'stride', 2, 'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [2 2], 'stride', 2, 'pad', 0);
net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [2 2], 'stride', 2, 'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [2 2], 'stride', 2, 'pad', 0);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
%% Add the first deconv layer
n_neuronsPre=size(HSI,3);
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool3', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x3', 'x6', 'deconv2f_pool3') ;
f = net.getParamIndex('deconv2f_pool3') ;

net.params(f).value = filters ;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool4', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x4', 'x7', 'deconv2f_pool4') ;
f = net.getParamIndex('deconv2f_pool4') ;
net.params(f).value = filters ;

net.params(f).value = filters ;
filters = single(bilinear_u(8, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool5', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 4, ...
    'crop', 2, ...
    'hasBias', false), ...
    'x5', 'x8', 'deconv2f_pool5') ;
f = net.getParamIndex('deconv2f_pool5') ;
net.params(f).value = filters ;
%% precious
net.vars(3).precious=true;
net.vars(4).precious=true;

net.featureName=datasets;
net.featurePath= 'data/features/multiscale_original';
net.eval({inputVar, single(HSI)}) ;
