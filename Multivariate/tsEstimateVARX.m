function [mu,A,B,SIGMA,U] = tsEstimateVARX(y,x,p,s,paramRestr)
% ------------------------------------------------------------------------------------
% Function to estimate VAR(p) model using Feasible Generalized Least Square
% ------------------------------------------------------------------------------------
% INPUT: y: K x 1 random variable vector, y(t) = [y(1,t)...y(K,t)]' k = 1...K, t = 1...T
%        x: K x 1 random variable vector, x(t) = [x(1,t)...x(M,t)]' m = 1...M, t = 1...T
%        p: number of lags (modelled variables)
%        s: number of lags (unmodelled variables)
%        paramRestr: parameter restrictions
% ------------------------------------------------------------------------------------
% OUTPUT: mu: mean vector
%         A: VAR(1) companion matrix (Kp + Ms) x (Kp + Ms)  
%         B: unmodelled variables matrix (Kp + Ms) x M
%         SIGMA: Covariance matrix (K x K)
%         U: residuals (K x T)
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
    x {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    s {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    paramRestr {mustBeNumeric} = []
end

% get dimensions
[t,K] = size(y);
[~,M] = size(x);


%-------------------------(2) Prepare dataset for estimation -------------------------

tmpYlags = tsMultMlag(y,p); % Y(t) = [y(t) ... y(t-p+1)]
Y = y(p + 1:t,:)'; % set start
Ylags = tmpYlags(p + 1:t,:)';

tmpXlags = tsMultMlag(x,s);
X = x(p + 1:t,:)';
Xlags = tmpXlags(p + 1:t,:)';

Z = [ones(1,t-p); Ylags; Xlags; X]; % Z(t) = [1 y(t) ... y(t-p+1)] (Ref.2 p.70)

%------------------------------(1) FGLS Estimation ------------------------------------

% Set Coeffs restrictions
R = eye(K*(1 + p*K + (s + 1)*M),K*(1 + p*K + (s + 1)*M));
if ~isempty(paramRestr)
    R(:,paramRestr) = [];
end

% First stage estimate using LS
AA = (Y*Z')/(Z*Z');
U = Y-AA*Z;
SIGMA = U*U'/(t-K*p-1);

% Second stage estimate the GLS
gamma = inv(R'*kron(Z*Z',inv(SIGMA))*R)*R'*(kron(Z,inv(SIGMA)))*vec(Y);
alpha = R*gamma; 
tmpA = reshape(alpha,K,(K*p + M*(s + 1) + 1));

% VAR(1) representation
mu = tmpA(:,1);
B0 = tmpA(:,end-M+1:end);
tmpAbig = [tmpA(:,2:K*p + M*s + 1); eye(K*(p-1)) zeros(K*(p-1),K)  zeros(K*(p-1),M*s)];
A = [tmpAbig; [zeros(M*s,K*p) [zeros(M,M*s); eye(M*(s-1)) zeros(M*(s-1),M)] ] ]; % companion matrix (Ref.2 p.403)
B = [B0 ; zeros(K*(p-1),M); eye(M*s,M)];

end