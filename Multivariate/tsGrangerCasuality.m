function [Wval,Fval,Wpval,Fpval,H] = tsGrangerCasuality(y,p,testParams,alpha,xParams)
% ------------------------------------------------------------------------------------
% Wald and F-test for Granger-Causality
% ------------------------------------------------------------------------------------
% INPUT: y: modelled (endogenous) variables 
%        p: number of lags of y
%        testParam: vector of parameters positions which wanted to be tested
%        alpha: sig level (default level: 0.05)
%        xParams.x (only for VARX specification): unmodelled (exogenous) variables 
%        xParams.s (only for VARX specification): number of lags of x
% ------------------------------------------------------------------------------------
% OUTPUT: WVal: result of the Wald test
%         FVal: result of the F-test
%         pval: p-value
%         H: hypothesis to accept
% ------------------------------------------------------------------------------------
% Refrences:
% 1. L.Kilian and H.Lütkepohl - Structural Vector Autoregressive Analysis (Oxford, 2017)
% 2. H.Lütkepohl - New Introduction to Multiple Time Series Analysis (Springer, 2005)
% 3. S.Hurn, V.Martin, D.Harris - Econometric Modelling with Time Series (Cambridge, 2012)
% 4. H.M.Pesaran - Time Series and Panel Data Econometrics (Oxford, 2015)
% ------------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% https://github.com/lkamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
% -----------------------------(1) check inputs, set env ------------------------------

arguments
    y {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    testParams double {mustBeNonempty,mustBeNumeric}
    alpha double {mustBeNonempty,mustBeNumeric} = 0.05
    xParams.x double = []
    xParams.s (1,1) double = 0
    xParams.B double = []
    xParams.M (1,1) double = 0
end

% Set variables
x = xParams.x;
s = xParams.s;
M = xParams.M;

% Get dimensions
[t,K] = size(y);
if ~isempty(x)
    [~,M] = size(x);
end

% Get the number of parameters to be tested
N = size(testParams,2);

% Set the degree of freedom
dof = N;

% Init constraints matrix
c = zeros(N,1);
C = zeros(N,K*(1 + K*p + (s + 1)*M));

% Set elements of C to 1 which want to be tested
for i=1:N
    C(i,testParams(:,i)) = 1;
end

% -----------------------------(2) calculate statistic --------------------------------

% Create vec(betahat)
if ~isempty(x)
    [mu,A,B,SIGMA,~,Z] = tsEstimateVARX(y,x,p,s);
    betahat = [mu(1:K,:) A(1:K,:) B(1:K,:)];
else
    [mu,A,SIGMA,~,Z] = tsEstimateVAR(y,p);
    betahat = [mu(1:K,:) A(1:K,:)];
end
betahat = reshape(betahat', [size(betahat,1)*size(betahat,2) 1])';

% Wald test
Wval = (C*betahat'-c)' * inv(C*(kron(inv(Z*Z'),SIGMA))*C') * (C*betahat'-c); 
% F-test
Fval = Wval/N; 
% p-value
Wpval = 1 - chi2cdf(Wval,dof);
Fpval = 1 - fcdf(Fval,N, K*t - ((K^2)*p) - K - (s + 1)*M);
% Hypothesis to accept
H = Fpval <= alpha;

end