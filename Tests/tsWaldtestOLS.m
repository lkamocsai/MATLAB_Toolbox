function [Wval,pval,H] = tsWaldtestOLS(theta1,t,H,R,Q,dof,alpha)
%-------------------------------------------------------------------------------
% Wald test
%
% H0: the parameter(s) = 0 (if p-value > alpha)
% H1: the parameter(s) not = 0 (if p-value < alpha)
%-------------------------------------------------------------------------------
% INPUT: theta: estimated parameters
%        t: time-series length
%        H: Hessian matrix
%        R: number of parameters
%        Q: restricted parameter value to test
%        dof: number of restrictions
%        alpha: sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: Wval: value of the test statistic
%         pval: p-value
%         H: hypothesis result
%-------------------------------------------------------------------------------

%--------------------(1) input check, base calculations ------------------------

switch nargin
    case 1
        error('input required: H, Q, K, dof, alpha (optional)')
    case 2
        error('input required: R, K, dof, alpha (optional)')
    case 3
        error('input required: Q, dof, alpha (optional)')
    case 4
        error('input required: dof, alpha (optional)')
    case 5
        error('input required: dof, alpha (optional)')
    case 6
        alpha = 0.05;
    case 7
    otherwise
        error('minimum input required: theta1, t, H, R, Q, dof, alpha')
end

% Covariance matrix
Omega = H; 

%--------------------(2) calculate statistic -----------------------------------

Wval = t*(R*theta1 - Q)'*inv(R*Omega*R')*(R*theta1 - Q);
pval = 1 - chi2cdf(Wval,dof);
H = pval <= alpha;

end