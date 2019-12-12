function xTilde=PCA_Reduction(Data,k)
%%
[row,colum,spec]=size(Data);
if spec~=1
    Data=(reshape(Data,row*colum,spec))';
end
Data = double(Data)./repmat(sqrt(sum(Data.^2,1)),size(Data,1),1);
%%
avg = mean(Data, 1); 
Data = Data - repmat(avg, size(Data, 1), 1);
sigma = Data * Data' / size(Data, 2);
[U,~,~] = svd(sigma);
%%
xTilde = U(:,1:k)' * Data; 
%%
xTilde=(xTilde-repmat((min(xTilde,[],2)),1,size(xTilde,2)))./(repmat((max(xTilde,[],2)),1,size(xTilde,2))-repmat((min(xTilde,[],2)),1,size(xTilde,2)));
xTilde=xTilde+abs(min(xTilde(:))).*255;
