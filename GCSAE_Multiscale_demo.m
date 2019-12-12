clear; close all; clc
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath SAE

dataname='paviaU'; % paviaU % salinas % indian_pines % KSC
featurePath='data/features';
param.featurename_spatial='multiscale_VGGspatial';
param.featurename_spectral='multiscale_original';
%% network parameters
param.hiddenSizeL1 = 50;  % paviaU 50 % salinas100 % indian_pines 100 KSC 80
param.hiddenSizeL2 = 100;
param.sparsityParam = [0.3,0.3]; % paviaU; salinas; indian_pines [0.3,0.3] % KSC [0.5,0.5]
param.lambda = [3e-2,3e-3]; % paviaU; salinas; indian_pines [3e-2,3e-3] KSC [3e-3,3e-3]
param.beta = 3;  % % paviaU; salinas; indian_pines 3 % KSC 5
param.layer=[4,38,4,37,3,36]; 
param.P=[5,4,3];
param.sumParam = [0.2,0.6];% paviaU 0.2,0.6 % salinas 0.3,0.4 % indian_pines 0.45,0.4 % SKC 0.4,0.25
param.alpha = 15; % paviaU 15 salinas 20 indian_pines 15 KSC 15
param.supersize=30;% paviaU:30;salinas 60 indian_pines 20 KSC 50
%% read features
F_spe=cell(4,length(param.layer)/2);
F_spa=cell(4,length(param.layer)/2);
k=1; block=cell(1,length(param.layer)/2);
for i=1:2:length(param.layer)
    spe_f=cell2mat(struct2cell(load(sprintf('%s/%s/%s_scale%d.mat',featurePath,param.featurename_spectral,dataname,param.layer(i)))));
    block{k}=local_consistency(spe_f,param.supersize);
    spa_f=cell2mat(struct2cell(load(sprintf('%s/%s/%s_scale%d.mat',featurePath,param.featurename_spatial,dataname,param.layer(i+1)))));
    spa_f=data_preprocessing_nonzero(spa_f);
    [F_spe{2,k},F_spe{3,k},F_spe{4,k}]=size(spe_f);
    [F_spa{2,k},F_spa{3,k},F_spa{4,k}]=size(spa_f);
    F_spe{1,k}=reshape(spe_f,F_spe{2,k}*F_spe{3,k},F_spe{4,k})';
    F_spa{1,k}=reshape(spa_f,F_spa{2,k}*F_spa{3,k},F_spa{4,k})';
    k=k+1;
end
%% feature fusion by CSAE
sae_Ffusion=cell(1,size(F_spa,2));
for i=1:size(F_spa,2)
    % fusion
    if i==3;param.lambda = [3e-3,3e-3];end
    sae_Ffusion{i}=stackedAE_fusionfeatures(F_spa{1,i},F_spe{1,i},block{i},param);
    for j=1:size(sae_Ffusion{i},1)
        spa_f_=sae_Ffusion{i}(j,:);
        map=reshape(spa_f_,F_spa{2,i},F_spe{3,i});
    end
end
%% fusion multiscale feature from L4 & L5 by weighted sum
sum_feature=param.sumParam(1)*sae_Ffusion{1,2}+(1-param.sumParam(1))*sae_Ffusion{1,1};
for j=1:size(sum_feature,1)
    spa_f_=sum_feature(j,:);
    map=reshape(spa_f_,F_spa{2,1},F_spe{3,1});
end

sum_feature=reshape(sum_feature',F_spa{2,1},F_spa{3,1},size(sum_feature,1));
%% upsampling
n_neuronsPre=size(sum_feature,3);
filters = single(bilinear_u(4, 1, n_neuronsPre)) ;
up_feature=vl_nnconvt(single(sum_feature), filters, [], ...
    'upsample', [2,2], ...
    'crop', [1,1,1,1], ...
    'numGroups', 1);
up_feature=reshape(up_feature,size(up_feature,1)*size(up_feature,2),size(up_feature,3))';
%% fusion multiscale feature from L3 & L4_L5 by weighted sum
sum_feature=param.sumParam(2)*sae_Ffusion{1,3}+(1-param.sumParam(2))*up_feature;
for j=1:size(sum_feature,1)
    spa_f_=sum_feature(j,:);
    map=reshape(spa_f_,F_spa{2,3},F_spe{3,3});
end

sum_feature=reshape(sum_feature',F_spa{2,3},F_spa{3,3},size(sum_feature,1));
%% upsampling
n_neuronsPre=size(sum_feature,3);
filters = single(bilinear_u(8, n_neuronsPre, n_neuronsPre)) ;
up_feature=vl_nnconvt(single(sum_feature), filters, [], ...
    'upsample', [4,4], ...
    'crop', [2,2,2,2], ...
    'numGroups', n_neuronsPre);

save(sprintf('stackedAE_feature/%s_multistacksum.mat',dataname), 'up_feature')