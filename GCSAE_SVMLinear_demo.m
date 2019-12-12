% Run SVM for a specific set of parameters without cross validation
clear; close all;clc;
addpath libsvm-3.21\matlab

datasetName{1}='paviaU';
number{1}=10;


datasetName{2}='salinas';
number{2}=10;

datasetName{3}='indian_pines';
number{3}=2;

datasetName{4}='KSC';
number{4}=10;

run_time=10;
param.featurePath_SAE='stackedAE_feature';
param.dataPath='data/HSI';

method_MSAE='multistacksum';
param.resultsPath='Results';
%% groud turth
ClassColor=[115,74,18;0,0,255;255,20,60;50,205,50;225,0,255;...
    128,0,128;255,105,180;0,255,255;255,255,0;255,215,0;138,43,226;...
    0,191,255;192,192,192;34,139,34;218,165,32;255,140,0;30,49,36;201,120,12;252,204,230;72,72,72]./255;

C=[10,1,10,10,10,10];

for i=1%:numel(datasetName)
    Data_gt=cell2mat(struct2cell(load(sprintf('%s/%s_gt.mat',param.dataPath,datasetName{i}))));
    HSI_fusion=double(cell2mat(struct2cell(load(sprintf('%s/%s_%s.mat',param.featurePath_SAE,datasetName{i},method_MSAE)))));
    
    if strcmp(datasetName{i},'paviaU')
        Data_gt=Data_gt(1:end-2,:,:);
        HSI_fusion=HSI_fusion(:,7:end-6,:);
    elseif strcmp(datasetName{i},'indian_pines')
        HSI_fusion=HSI_fusion(8:end-8,8:end-8,:);
    elseif strcmp(datasetName{i},'salinas')
        HSI_fusion=HSI_fusion(:,4:end-4,:);
    elseif strcmp(datasetName{i},'KSC')
        Data_gt=Data_gt(:,4:end-3,:);
    end
    
    [row,colum,band]=size(HSI_fusion);
    labels=double((reshape(Data_gt,row*colum,1)));
    HSI_fusion=double(reshape(HSI_fusion,row*colum,band)');
    for num=1:length(number{i})
        %% SVM
        param.C=C(i);
        
        bestResult=0;Result.CA=zeros(max(max(Data_gt)),run_time);
        Result.AA=zeros(1,run_time);Result.OA=zeros(1,run_time);Result.kappa=zeros(1,run_time);
        
        for run=1:run_time
            Train_sample=load(sprintf('data/Trainsamples/%s_%d_%d.mat',datasetName{i},number{i}(num),run));
            datasets=Train_sample.datasets;
            tic;
            %% SVM Learning
            model = libsvmtrain(labels(datasets.trainIndex),HSI_fusion(:,datasets.trainIndex)',sprintf('-q -t 0 -c %f',param.C));
            Result.time(1,run)=toc;
            %% Predict test labels
            tic
            [tlabs,p1,p2] = libsvmpredict(labels(datasets.testIndex),HSI_fusion(:,datasets.testIndex)', model);
            Result.time(2,run)=toc;
            tlabs_unlabed = libsvmpredict(zeros(length(datasets.unLabledIndex),1),HSI_fusion(:,datasets.unLabledIndex)', model);
            
            ID=[datasets.trainIndex;datasets.testIndex;datasets.unLabledIndex];
            pre_label=[labels(datasets.trainIndex);tlabs;tlabs_unlabed];
            labelmap=zeros(length(ID),1);
            labelmap(ID)=pre_label;
            labelmap=reshape(labelmap,datasets.size(1),datasets.size(2));
            PredictGroudTruth=label2rgb(uint8(labelmap),ClassColor);
%             figure;imshow(PredictGroudTruth);
%             impixelinfo
            [Result.kappa(run),Result.CA(:,run),Result.OA(run),Result.AA(run)] = evaluate_results(tlabs, labels(datasets.testIndex));
            
            if Result.AA(run)>bestResult
                imwrite(PredictGroudTruth,sprintf('%s/%s_NumPerClass%d.png',param.resultsPath,datasetName{i},number{i}(num)))
                bestResult=Result.AA(run);
            end
            fprintf('Datasets:%s number:%d AA:%1.2f run:%d/%d \n',datasetName{i},number{i}(num),Result.AA(run)*100,run,run_time)
        end
        save(sprintf('%s/Accuracy/%s_NumPerClass%d.mat',param.resultsPath,datasetName{i},number{i}(num)),'-struct', 'Result')
    end
end