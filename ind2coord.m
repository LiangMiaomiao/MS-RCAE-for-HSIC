function y=ind2coord(x,n)
y=zeros(length(x),2);
for j=1:length(x)
    y(j,2)=1+fix((x(j)-0.001)/n);
    y(j,1)=rem(x(j),n);
    if y(j,1)==0
        y(j,1)=n;
    end
end