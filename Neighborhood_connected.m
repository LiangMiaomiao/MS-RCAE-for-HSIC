function sf=Neighborhood_connected(rr,m)
sf=[];
rt=[rr(1)+1,rr(1)+m-1,rr(1)+m,rr(1)+m+1];
for j=2:length(rr)
    if rr(j)==1
        sf=[sf,rt(j-1)];
    end
end
sf=[rr(1),sf];