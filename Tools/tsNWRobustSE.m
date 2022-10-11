function [NWRobustVCV, NWRobustSE] = tsNWRobustSE(X,uhat,nlag)
%-------------------------------------------------------------------------------
% Newey-West HAC robust estimator
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

[T,~] = size(X);

XXinv = inv(X'*X/T);

V = bsxfun(@times,X,uhat);

w=(nlag+1-(0:nlag))./(nlag+1);

VV = V'*V/T;

for i = 1:nlag
    Gammai = (V((i+1):T,:)' * V(1:T-i,:))/T;
    VV = VV + w(i+1) * (Gammai + Gammai');
end


NWRobustVCV = XXinv * VV * XXinv/T;
NWRobustSE = sqrt(diag(NWRobustVCV));

end
