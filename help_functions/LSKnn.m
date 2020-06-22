function NN=LSKnn(X1,X2,ks)
B=750;
[D,N]=size(X2);
NN=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
%   fprintf('.');
 Dist=distance(X1,X2(:,i:i+BB));
%   fprintf('.');
%   fprintf('.');
  [dist,nn]=mink(Dist,max(ks));
  clear('Dist');
%   fprintf('.');

%The following 'if' argument is added in the case that some class has a very small number of instances
if size(nn,1)-1 < length(ks)
    nn = [nn;repmat(nn(1,:),length(ks)-size(nn,1)+1,1)];
end
  NN(:,i:i+BB)=nn(ks,:);%change: nn(ks,:) to nn(ks(1:BB),:);

  clear('nn','dist');
%   fprintf('(%i%%) ',round((i+BB)/N*100));
end

end

