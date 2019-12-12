clear;clc;

datasetName{1}='paviaU';
number{1}=10;

datasetName{2}='salinas';
number{2}=10;

datasetName{3}='indian_pines';
number{3}=2;

datasetName{4}='KSC';
number{4}=10;

run_time=10;

for i=1%:numel(datasetName)
    Data_gt=cell2mat(struct2cell(load(sprintf('data/HSI/%s_gt.mat',datasetName{i}))));
    if strcmp(datasetName{i},'paviaU')
        Data_gt=Data_gt(1:end-2,:,:);
    elseif strcmp(datasetName{i},'KSC')
        Data_gt=Data_gt(:,4:end-3,:);
    end
    for k=1:length(number{i})
        for j=1:run_time
            datasets=HSI_trainDataCreation(Data_gt,number{i}(k));
            save(sprintf('data/Trainsamples/%s_%d_%d.mat',datasetName{i},number{i}(k),j), 'datasets')
        end
    end
end
