function [LRval,pval,H] = tsLRtest(logL0,logL1,dof,alpha)
%-------------------------------------------------------------------------------
% Likelihood Ratio test
%
% H0: the parameter(s) = 0 (if p-value > alpha)
% H1: the parameter(s) not = 0 (if p-value < alpha)
%-------------------------------------------------------------------------------
% INPUT: logL0: log-likelihood of the restricted model
%        logL1: log-likelihood of the unrestricted model
%        dof: number of restrictions
%        alpha: sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: LRval: value of test statistic
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
        error('input required: logL1, dof, alpha (optional)')
    case 2
        error('input required: dof, alpha (optional)')
    case 3
        alpha = 0.05;
    case 4
    otherwise
        error('minimum input required: logL0, logL1, dof')
end

%--------------------(2) calculate statistic -----------------------------------
LRval = -2*(logL1 - logL0);
pval = 1 - chi2cdf(LRval,dof);
H = pval <= alpha;

end

