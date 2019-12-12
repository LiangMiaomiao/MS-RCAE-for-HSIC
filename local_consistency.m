function connected_components=local_consistency(datas,unconn_Patch_Num)
addpath GCmex2.0;
%% original image
[row,colum,spec]=size(datas);
datasets=reshape(datas,row*colum,spec)';
%% PCA Reduction
principal_Num=3;
xTilde=PCA_Reduction(datasets,principal_Num);
im=reshape(xTilde',row,colum,principal_Num);
%% Éú³É³¬ÏñËØ
% unconn_Patch_Num=30;% % try to segment the image into k different regions
% color space distance
distance = 'sqEuclidean';
% cluster the image colors into k regions
data = ToVector_K_dimention(im,principal_Num);
[idx, c] = kmeans(data, unconn_Patch_Num, 'distance', distance,'maxiter',500);

% calculate the data cost per cluster center
Dc = zeros([row,colum,unconn_Patch_Num],'single');
for ci=1:unconn_Patch_Num
    % use covariance matrix per cluster
    icv = pinv(cov(data(idx==ci,:)));
    dif = data - repmat(c(ci,:), [size(data,1) 1]);
    % data cost is minus log likelihood of the pixel to belong to each
    % cluster according to its RGB value
    Dc(:,:,ci) = reshape(sum((dif*icv).*dif./2,2),row,colum);
end
%% cut the graph
% smoothness term:
% constant part
Sc = ones(unconn_Patch_Num) - eye(unconn_Patch_Num);
% spatialy varying part
% [Hc Vc] = gradient(imfilter(rgb2gray(im),fspecial('gauss',[3 3]),'symmetric'));
[Hc, Vc] = SpatialCues(im);

gch = GraphCut('open', Dc, 10*Sc, exp(-Vc*5), exp(-Hc*5));
[gch, L] = GraphCut('expand',gch);
gch = GraphCut('close', gch);
L=L+1;
Superpixel=double((reshape(L,row*colum,1))');
%% Unconnected superpixel
Data_SuperPatch_unconn=zeros(row*colum,3);
Data_SuperNum_uncon=zeros(row*colum,1);  
unconn_PatchColo=randi([1,255],unconn_Patch_Num,3);
for i=1:unconn_Patch_Num
    Super_i=find(Superpixel==i);
    Data_SuperPatch_unconn(Super_i,:)=repmat(unconn_PatchColo(i,:),length(Super_i),1);
    Data_SuperNum_uncon(Super_i)=i;
end
Data_SuperNum_uncon=reshape(Data_SuperNum_uncon,row,colum);
%% connected superpixel by Neighborhood connected
W=zeros(row*colum,6);
W(1,1:4)=1;
for j=1:colum-1
    for i=1:row-1
        k=coord2ind(i,j,row);
        W(k,1)=k;
        if i==1
            for i1=[6,8,9]
                y=ind2coord(i1,3);
                W(k,i1-4)=Data_SuperNum_uncon(i,j)==Data_SuperNum_uncon(y(1)+i-2,y(2)+j-2);
            end
        else
            for i1=6:9
                y=ind2coord(i1,3);
                W(k,i1-4)=Data_SuperNum_uncon(i,j)==Data_SuperNum_uncon(y(1)+i-2,y(2)+j-2);
            end
        end
        W(k,6)=1;
    end
end
%% connected superpixels
connected_trans=[];crack_connected=zeros(row,colum);
crack_conncted_SC=zeros(row,colum);L_conn=zeros(row*colum,1);
w=1;
for i=1:length(W(:,1))
    if W(i,6)~=0
        connected_neigh=Neighborhood_connected(W(i,1:5),row);
        W(i,6)=0;
        [new_componet,W]=Transitivity_Connectivity(connected_neigh,W,row);
        connected_trans=[new_componet,connected_trans];
        while ~isempty(new_componet)
            [new_componet,W]=Transitivity_Connectivity(new_componet,W,row);
            connected_trans=[new_componet,connected_trans];
        end
        for j=connected_trans
            y=ind2coord(j,row);
            crack_connected(y(1),y(2))=1;
        end
        connected_components{w}=connected_trans;
        connected_components{w}(end+1)=i;
        L_conn(connected_components{w})=w;
        w=w+1;
    end
    connected_neigh=[];connected_trans=[];
end
L_conn=reshape(L_conn,row,colum);
%% 
All_conn_Patch_Num=numel(connected_components);
Data_SuperPatch_conn=zeros(row*colum,3);
conn_PatchColo=randi([1,255],All_conn_Patch_Num,3);
for i=1:All_conn_Patch_Num
    Data_SuperPatch_conn(connected_components{i},:)=repmat(conn_PatchColo(i,:),length(connected_components{i}),1);
end
figure;imshow(uint8(reshape(Data_SuperPatch_conn,row,colum,3)));
impixelinfo
