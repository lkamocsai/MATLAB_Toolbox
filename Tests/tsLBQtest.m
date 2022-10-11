function [lbqval,pval,H] = tsLBQtest(ts,k,alpha)
%-------------------------------------------------------------------------------
% Ljung-Box Q-statistic to test autocorrelation in the residuals
%
% H0: no autocorrelation in the residuals (if p-value > alpha)
% H1: autocorrelation in the residuals (if p-value < alpha)
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        k: autocorrelation lag length
%        alpha: sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: lbqout: value of test statistic
%         pval: p-value
%         H: hypothesis result
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%--------------------(1) input check, base calculations ------------------------
switch nargin
    case 1
        k = 5;
        alpha = 0.05;
    case 2
        alpha = 0.05;
    case 3
    otherwise
        error('minimum input required: time series vector')
end

T = length(ts);
rho = autocorr(ts,'NumLags',k);
rho = rho(2:end);
tmp = (rho.^2)./(T-(1:k))';

%--------------------(2) calculate statistic -----------------------------------
lbqval = (T*(T+2))*(sum(tmp(1:end)));
pval = 1-chi2cdf(lbqval,k);
H = pval < alpha;

end