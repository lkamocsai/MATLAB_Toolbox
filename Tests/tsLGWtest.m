function [ntest,pval] = tsLGWtest(y,X,theta,neurons)

tmpg(:,1) = theta(7:10)';
tmpg(:,2) = theta(11:14)';
g = tmpg;
    for j = 1:neurons
        lambda = X*g(:,j);
        w(:,j) = log( 1 + exp(lambda) );
    end

    uhat = y - X*(X\y); % residuals from regression y on X
    tmpXw = [X w];
    uuhat = uhat - tmpXw*(tmpXw\uhat); % regress residuals on X and w
    ntest = size(y,1)*(1 - sum(uuhat.^2)/sum(uhat.^2)); % T * R2
    pval    =  1 - chi2cdf(ntest,neurons);
    pred = tmpXw*(tmpXw\y);
    
end