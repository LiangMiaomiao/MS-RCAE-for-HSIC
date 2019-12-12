function data=data_preprocessing(data)
[row,colum,fea]=size(data);
if fea~=1
    data=reshape(data,row*colum,fea)';
    data=(data-repmat(mean(data,2),1,row*colum))./repmat(std(data,0,2),1,row*colum);
    data=reshape(data',row,colum,fea);
else
    data=(data-repmat(mean(data,[],2),1,colum))./repmat(std(data,0,2),1,colum);
end