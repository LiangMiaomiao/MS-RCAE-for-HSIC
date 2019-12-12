function data=data_preprocessing_nonzero(data)
[row,colum,fea]=size(data);
if fea~=1
    data=reshape(data,row*colum,fea)';
    for i=1:fea
        location=find(abs(data(i,:))>=1e-10); % 1e-10
        %% z-score
        mean_nunzero=mean(data(i,location));
        std_nunzero=std(data(i,location));
        data(i,location)=(data(i,location)-mean_nunzero)./ std_nunzero;
    end
    data=reshape(data',row,colum,fea);
else
    for i=1:size(data,1)
        location=find(abs(data(i,:))>1e-10);
        %% z-score
        mean_nunzero=mean(data(i,location));
        std_nunzero=std(data(i,location));
        data(i,location)=(data(i,location)-mean_nunzero)./ std_nunzero;
    end
end