function [LMval,pval,H] = tsARCHLMtest(ts,k,alpha)
%-------------------------------------------------------------------------------
% ARCH LM statistic to test heteroscedastic
%
% H0: no ARCH, the ts homoscedastic (if p-value > alpha)
% H1: ARCH, the ts heteroscedastic (if p-value < alpha)
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        k: autocorrelation lag length
%        alpha: sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: LM: value of test statistic
%         pval: p-value
%         H: hypothesis result
%-------------------------------------------------------------------------------

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

ts = ts.^2;
ts_klag = tstrim(tsklagmat(ts,k,1),k,0);
T = length(ts_klag);

%--------------------(2) auxiliary regression -----------------------------------
M0 = eye(T)-1/(T)*(ones(T,1)*ones(T,1)');
b = ts_klag\tstrim(ts,k,0);
e_hat = tstrim(ts,k,0) - ts_klag * b;
R2 = 1 - ((e_hat'*e_hat)/(tstrim(ts,k,0)' * M0 * tstrim(ts,k,0)));

%--------------------(3) calculate statistic -----------------------------------
LMval = T * R2;
pval = 1 - chi2cdf(LMval,k);
H = pval < alpha;

end