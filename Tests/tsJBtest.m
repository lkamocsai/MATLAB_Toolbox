function [tsjbval,pval,H] = tsJBtest(ts,alpha)
%-------------------------------------------------------------------------------
% Test time series normality, Jarque-Bera test
%
% H0: the distribution is normal (if p-value > alpha)
% H1: the distribution is non-normal (if p-value < alpha)
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        alpha: sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: tsjbval: value of test statistic
%         pval: p-value
%         H: hypothesis result
%-------------------------------------------------------------------------------

%--------------------(1) input check, base calculations ------------------------
switch nargin
    case 1
        alpha = 0.05;
    case 2
    otherwise
        error('minimum input required: time series vector')
end

T = length(ts);
S = skewness(ts);
K = kurtosis(ts);

%--------------------(2) calculate statistic -----------------------------------
tsjbval = (T/6)*(S^2 + (((K-3)^2)/4) );
pval = 1 - chi2cdf(tsjbval,2);
H = pval < alpha;

end