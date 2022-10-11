function nLL = tsLLnormal(y,X,theta)
%-------------------------------------------------------------------------------
% Log-likelihood function assuming normal density
%
%-------------------------------------------------------------------------------
% INPUT: y: T-by-1 vector of dependent variables
%        X: T-by-K matrix of independent variables
%        theta: 1-by-K vector of parameters to be estimated
%-------------------------------------------------------------------------------
% OUTPUT: nLL: value of test statistic        
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

%--------------------(1) input check, base calculations ------------------------
switch nargin
    case 1
        error('input missing: X, theta')
    case 2
        error('input missing: theta')
    case 3
    otherwise
        error('required inputs: y, X, theta')
end

[T,K] = size(X);

beta = theta(1:K);
sigma = theta(K + 1);
u_t = y - X * beta';

%-------------------(2) calculate log-likelihood function-----------------------

LL = -(T/2) * log(2*pi) -(T/2) * log(sigma^2) - (1/(2 * sigma^2)) * (u_t'*u_t);
nLL = -LL;

end