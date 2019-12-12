function [xin,W]=Transitivity_Connectivity(ss,W,m)
xin=[];
for j=ss
    if W(j,6)~=0
        sz=Neighborhood_connected(W(j,1:5),m);
        W(j,6)=0;
        xin=[xin,sz];
    end
end