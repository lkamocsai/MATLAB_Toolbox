function [mu,A,B,SIGMA,U,Z,tvals,pvals] = tsEstimateVARX(y,x,p,s,paramRestr)
% ------------------------------------------------------------------------------------
% Function to estimate VARX(p,s) model using Feasible Generalized Least Square
% ------------------------------------------------------------------------------------
% INPUT: y: K x 1 random (endogenous) variable vector, y(t) = [y(1,t)...y(K,t)]' k = 1...K, t = 1...T
%        x: K x 1 random (exogenous) variable vector, x(t) = [x(1,t)...x(M,t)]' m = 1...M, t = 1...T
%        p: number of lags (modelled variables)
%        s: number of lags (unmodelled variables)
%        paramRestr: parameter restrictions
% ------------------------------------------------------------------------------------
% OUTPUT: mu: mean vector
%         A: VAR(1) companion matrix (Kp + Ms) x (Kp + Ms)  
%         B: unmodelled variables coefficient matrix (Kp + Ms) x M
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
% https://github.com/lkamocsai
% lkamocsai@student.elte.hu
% Version: 1.1    Date: 29/10/2022
%
% ------------------------------------------------------------------------------------
%
% -----------------------------(1) check inputs, set env -----------------------------

arguments
    y {mustBeNonempty,mustBeNumeric}
    x {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    s {mustBeNonempty,mustBeNumeric,mustBeLessThanOrEqual(s,p) }
    paramRestr {mustBeNumeric} = []
end

% get dimensions
[T,K] = size(y);
[~,M] = size(x);

% -----------------------------(2) Prepare dataset for estimation --------------------

tmpYlags = tsMultMlag(y,p); % Y(t) = [y(t) ... y(t-p+1)]
Y = y(p + 1:T,:)'; % set start
Ylags = tmpYlags(p + 1:T,:)';

% Check if lagged exogenous variable considered or not
if s ~= 0
tmpXlags = tsMultMlag(x,s);
X = x(p + 1:T,:)';
Xlags = tmpXlags(p + 1:T,:)';
else 
    X = x(p + 1:T,:)';
    Xlags = [];
end

Z = [ones(1,T-p); Ylags; Xlags; X]; % Z(t) = [1 y(t) ... y(t-p+1)] (Ref.2 p.70)

% -----------------------------(3) FGLS Estimation -----------------------------------

% Set Coeffs restrictions
R = eye(K*(1 + p*K + (s + 1)*M),K*(1 + p*K + (s + 1)*M));
if ~isempty(paramRestr)
    R(:,paramRestr) = [];
end

% First stage estimate using LS
AA = (Y*Z')/(Z*Z');
U = Y-AA*Z;
SIGMA = U*U'/(T-K*p-1);

% Second stage estimate the GLS
gamma = inv(R'*kron(Z*Z',inv(SIGMA))*R) * R' *(kron(Z,inv(SIGMA))) * vec(Y);
alpha = R*gamma; 
tmpA = reshape(alpha,K,(K*p + M*(s + 1) + 1));

% -----------------------------(4) VAR(1) representation -----------------------------

mu = tmpA(:,1);
B0 = tmpA(:,end-M+1:end);
tmpAbig = [tmpA(:,2:K*p + M*s + 1); eye(K*(p-1)) zeros(K*(p-1),K)  zeros(K*(p-1),M*s)];
A = [tmpAbig; [zeros(M*s,K*p) [zeros(M,M*s); eye(M*(s-1)) zeros(M*(s-1),M)] ] ]; % companion matrix (Ref.2 p.403)
B = [B0 ; zeros(K*(p-1),M); eye(M*s,M)];

% -----------------------------(5) Test coeffs ---------------------------------------

dof = T - K*(p-1) - M*(s-1);
vectmpA = vec(tmpA);
tvals = vectmpA./sqrt(diag(kron(inv(Z*Z'),SIGMA)));
pvals = zeros(size(tvals,1),1);
for i = 1:size(tvals,1)
    pvals(i,:) = 2*(1-tcdf(abs(tvals(i)),dof));
end 
tvals = round( reshape(tvals,size(tmpA,1),size(tmpA,2)),3);
pvals = round( reshape(pvals,size(tmpA,1),size(tmpA,2)),3);

end