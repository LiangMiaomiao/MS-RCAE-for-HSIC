function block_size= block_err(block)
a=zeros(1,numel(block));
for i=1:numel(block)
    a(i)=length(block{i});
end
block_size=sum(a>10); % 100