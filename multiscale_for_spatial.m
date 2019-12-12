run matconvnet/matlab/vl_setupnn ;
addpath matconvnet
addpath data;
%% experiment setup
datasets = 'paviaU';% paviaU; salinas; indian_pines; KSC
modelName='imagenet-vgg-verydeep-16';

opts.expDir = 'data/fcn32s' ;
imageNeedsToBeMultiple=1;
opts.modelPath = sprintf('data/models/%s.mat',modelName);
%% Setup model
net = load(opts.modelPath) ;
for layers=35:numel(net.layers)
    net.layers{layers}={};
end
net.layers(cellfun(@isempty,net.layers))=[];
net.layers{32}.pad=[3,3,3,3];
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
%% add deconv layer
n_neuronsPre=256;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool3', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x17', 'pool3_up2', 'deconv2f_pool3') ;
f = net.getParamIndex('deconv2f_pool3') ;
net.params(f).value = filters ;

n_neuronsPre=512;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool4', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x24', 'pool4_up2', 'deconv2f_pool4') ;
f = net.getParamIndex('deconv2f_pool4') ;
net.params(f).value = filters ;

n_neuronsPre=512;
filters = single(bilinear_u(8, 1, n_neuronsPre)) ;
net.addLayer('deconv2_pool5', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 4, ...
    'crop', 2, ...
    'hasBias', false), ...
    'x31', 'pool5_up4', 'deconv2f_pool5') ;

f = net.getParamIndex('deconv2f_pool5') ;
net.params(f).value = filters ;

n_neuronsPre=512;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_poolup1', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x31', 'pool5_up2_1', 'deconv2f_poolup1') ;

f = net.getParamIndex('deconv2f_poolup1') ;
net.params(f).value = filters ;

net.params(f).value = filters ;
n_neuronsPre=512;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_poolup2', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'pool5_up2_1', 'pool5_up2_2', 'deconv2f_poolup2') ;

f = net.getParamIndex('deconv2f_poolup2') ;
net.params(f).value = filters ;

net.params(f).value = filters ;
n_neuronsPre=512;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_convup', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x29', 'pool5_convup2', 'deconv2f_convup') ;

f = net.getParamIndex('deconv2f_convup') ;
net.params(f).value = filters ;

net.params(f).value = filters ;
n_neuronsPre=512;
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
net.addLayer('deconv2_reluup', ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', 2, ...
    'crop', 1, ...
    'hasBias', false), ...
    'x30', 'pool5_reluup2', 'deconv2f_reluup') ;

f = net.getParamIndex('deconv2f_reluup') ;
net.params(f).value = filters ;
 
net.vars(36).precious=true; 
net.vars(37).precious=true; 
net.vars(38).precious=true; 
%% Data pretreat
HSI=double(cell2mat(struct2cell(load(sprintf('data/HSI/%s.mat',datasets)))));
[row,colum,~]=size(HSI);
xTilde=PCA_Reduction(HSI,3);
HSI=(reshape(xTilde',row,colum,3));
Data=(reshape(HSI,row*colum,3))';

if max(HSI(:))<=1; HSI=HSI*255; end
if imageNeedsToBeMultiple
    sz = [size(HSI,1), size(HSI,2)] ;
    sz_ = round(sz / 32)*32 ;
    HSI = imresize(HSI, sz_,'nearest') ;
end

HSI=single(HSI);
average(:,:,1)=mean(mean(HSI(:,:,1)));
average(:,:,2)=mean(mean(HSI(:,:,2)));
average(:,:,3)=mean(mean(HSI(:,:,3)));
im = bsxfun(@minus, HSI, average) ; 
%% feature extracting
inputVar = 'input' ;
net.featureName=datasets;
net.featurePath='data/features/multiscale_VGGspatial'; 
net.eval({inputVar, single(im)}) ;