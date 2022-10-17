function [mu,A,SIGMA,U,Z] = tsEstimateVAR(y,p,paramRestr)
% ------------------------------------------------------------------------------------
% Function to estimate VAR(p) model using Feasible Generalized Least Square
% ------------------------------------------------------------------------------------
% INPUT: y: K x 1 random variable vector, y(t) = [y(1,t)...y(K,t)]' k = 1...K, t = 1...T
%        p: number of lags
%        paramRestr: parameter restrictions
% ------------------------------------------------------------------------------------
% OUTPUT: mu: mean vector
%         A: VAR(1) companion matrix (Kp x Kp)
%         SIGMA: Covariance matrix (K x K)
%         U: residual (K x T)
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
% ------------------------------------------------------------------------------------
%
% -----------------------------(1) check inputs, set env -----------------------------

arguments
    y {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    paramRestr {mustBeNumeric} = []
end

% Get dimensions
[t,K] = size(y);

% -----------------------------(2) Prepare dataset for estimation --------------------

tmpYlags = tsMultMlag(y,p); % Y(t) = [y(t) ... y(t-p+1)]
Y = y(p + 1:t,:)'; % set start
Ylags = tmpYlags(p + 1:t,:)';
Z = [ones(1,t-p); Ylags]; % Z(t) = [1 y(t) ... y(t-p+1)] (Ref.2 p.70)

% -----------------------------(3) FGLS Estimation -----------------------------------

% Set Coeffs restrictions
R = eye(K*(1 + p*K),K*(1 + p*K)); 
if ~isempty(paramRestr)
    R(:,paramRestr) = [];
end

% First stage estimate using LS
AA = (Y*Z')/(Z*Z');
U = Y-AA*Z;
SIGMA = U*U'/(t-K*p-1);

% Second stage estimate the GLS
gamma = inv(R'*kron(Z*Z',inv(SIGMA))*R) * R' * (kron(Z,inv(SIGMA))) * vec(Y);
alpha = R*gamma; 
tmpA = reshape(alpha,K,(K*p + 1));

% -----------------------------(4) VAR(1) representation -----------------------------

mu = [tmpA(1:K,1); zeros(K,1)];
A = tmpA(:,2:(K*p) + 1);
A = [A(1:K,:);eye(K*(p-1)) zeros(K*(p-1),K)]; % companion matrix (Ref.2 p.15)

end