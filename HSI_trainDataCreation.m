function datasets=HSI_trainDataCreation(Data_gt,samplePerClass)
[row,colum]=size(Data_gt);
Class_Numble=max(max(Data_gt));
%% sample Per Class
if samplePerClass>=1
    if samplePerClass==1
        numPerClass=[5,143,83,23,50,75,3,49,2,97,247,61,21,129,38,10]; %% for Indian Pines
    elseif samplePerClass==2
        numPerClass=[3,72,42,12,25,38,2,25,2,49,124,31,11,65,19,5]; %% each class with different number of training samples
    else
        numPerClass=samplePerClass*ones(1,Class_Numble); %% each class with same number of training samples
    end
else
    numPerClass=samplePerClass;  % precent
end
%% Ground_truth map
class_index=cell(Class_Numble,1); 
class_index(1)={find(Data_gt==0)};
each_class_number=zeros(1,Class_Numble);

for i=1:Class_Numble
    index=find(Data_gt==i);
    class_index(i+1)={index};
    each_class_number(i)=length(index);
end
%% training and testing
H_train=[];train_index=[];
H_test=[];test_index=[];

for k=1:Class_Numble
    if length(numPerClass)==1
        % precent
        train_pixel=randperm(each_class_number(k),fix(each_class_number(k)*numPerClass));
        test_pixel=setdiff(1:each_class_number(k),train_pixel);
    elseif length(numPerClass)>1
        % given number
        train_pixel=randperm(each_class_number(k),numPerClass(k));
        test_pixel=setdiff(1:each_class_number(k),train_pixel);
    end
    
    h_train=zeros(Class_Numble,length(train_pixel));h_train(k,:)=1;
    H_train=[H_train,h_train];
    train_index=[train_index,class_index{k+1}(train_pixel)'];
    
    h_test=zeros(Class_Numble,length(test_pixel));h_test(k,:)=1;
    H_test=[H_test,h_test];
    test_index=[test_index,class_index{k+1}(test_pixel)'];
end

unlabel_indix=class_index{1}';

datasets.size=[row,colum];
datasets.trainIndex=train_index';
datasets.testIndex=test_index';
datasets.unLabledIndex=unlabel_indix';
datasets.H_train=H_train;
datasets.H_test=H_test;
