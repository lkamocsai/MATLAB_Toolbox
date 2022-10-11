function [mu,A,SIGMA,U] = tsEstimateVAR(y,p,paramRestr)
% ------------------------------------------------------------------------------------
% Function to estimate VAR(p) model using Feasible Generalized Least Square
% ------------------------------------------------------------------------------------
% INPUT: y: K x 1 random variable vector, y(t) = [y(1,t)...y(K,t)]' k = 1...K, t = 1...T
%        p: number of lags
%        VARmode: enter 'VAR1' if one want to use VAR(1) representation of
%        VAR(p) model (default value: 'VARp')
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
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-----------------------------(1) check inputs, set env ------------------------------

arguments
    y {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    paramRestr {mustBeNumeric} = []
end

% get dimensions
[t,K] = size(y);

%-------------------------(2) Prepare dataset for estimation -------------------------

lags = tsMultMlag(y,p); % Y(t) = [y(t) ... y(t-p+1)]
Y = y(p + 1:t,:)'; % set start
Ylags = lags(p + 1:t,:)';
Z = [ones(1,t-p); Ylags]; % Z(t) = [1 y(t) ... y(t-p+1)] (Ref.2 p.70)

%------------------------------(1) FGLS Estimation ------------------------------------

% Set Coeffs restrictions
R = eye(K*(1 + p*K),K*(1 + p*K)); 
if ~isempty(paramRestr)
    R(:,paramRestr) = [];
end

% Estimate Coefficients
gamma = inv(R'*kron(Z*Z',eye(K))*R)*R'*(kron(Z,eye(K)))*vec(Y);
alpha = R*gamma; 
A = reshape(alpha,K,(K*p + 1)); 

% Estimate consistent Covariance matrix
u = vec(Y) - kron(Z',eye(K))*R*gamma;
U = reshape(u,K,size(Y,2));
SIGMA = U*U'/(t); % 

% VAR(1) representation
mu = [A(1:K,1); zeros(K,1)];
A = A(:,2:(K*p) + 1);
A = [A(1:K,:);eye(K*(p-1)) zeros(K*(p-1),K)]; % Kp x Kp dimensional companion matrix (Ref.2 p.15)

end