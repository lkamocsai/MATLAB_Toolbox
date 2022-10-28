function [P,PHI,IRFOrth,IRFGen,D] = tsIRF(A,SIGMA,p,h,method,xParams)
% ------------------------------------------------------------------------------------
% Function to estimate MA(oo) coefficient matrices and h-step Inpulse Response Functions
% ------------------------------------------------------------------------------------
% INPUT: A: VAR coefficient matrix (Kp x Kp)
%        SIGMA: Covariance matrix (K x K)
%        p: number of lags
%        h: number of steps ahead 
%        xParams.s (only for VARX specification): number of unmodelled variable lags
%        xParams.M (only for VARX specification): number of unmodelled variables
%        xParams.B (only for VARX specification): exogenous variables coefficient matrix ((Kp + Ms) x M)
% ------------------------------------------------------------------------------------
% OUTPUT: P: lower-triangular Choleski matrix (K x K)
%         PHI: MA weight matrices (K x K x h)
%         IRFOrth: Orthogonalized Inpulse Response (K x K x h)
%         IRFGen: Generalized Inpulse Response (K x K x h)
%         D: Dynamic multiplier
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
% ------------------------------(1) check inputs -------------------------------------

arguments
    A {mustBeNonempty,mustBeNumeric}
    SIGMA {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    h {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    method char {mustBeMember(method,{'IRFOrth','IRFGen'})} = 'IRFGen'
    xParams.s (1,1) double = 0
    xParams.M (1,1) double = 0
    xParams.B double = []
end

% ------------------------------(2) set env ------------------------------------------

K = size(SIGMA,1);
J = [eye(K) zeros(K, (K*p) + (xParams.M*xParams.s) - K)]; % (Ref.1 p.26, Ref.2 p.403)
PHI = zeros([K K h+1]);
e = eye(K); % selection vector
IRFOrth = zeros(K,K,h);
IRFGen = zeros(K,K,h);

% ------------------------------(3) MA representation of VAR(p) ----------------------

P = chol(SIGMA)'; % Choleski decomposition of SIGMA = P * P', (Ref.1 p.109)
PHI(:,:,1) =J * (A^0) * J'; % PHI(i) = J * A^i * J' where PHI(0) = IK (Ref.1 p.111, Ref.2 p.18)
for i = 1:h
    PHI(:,:,i+1) = J * (A^i) * J';
end

% ------------------------------(4) Dynamic multipliers --------------------------------

if ~isempty(xParams.B)
D(:,:,1) = J * (A^0) * xParams.B; % (Ref.2 p.407)
    for i = 1:h-1
        D(:,:,i+1) = J * (A^i) * xParams.B;
    end
else
    D = 0;
end

% ------------------------------(5) Inpulse Responses --------------------------------

for hh = 1:h
    for ii = 1:K
        if method == 'IRFOrth'
        % IRF Orthogonalized (Ref.5 p.586)
        IRFOrth(:,ii,hh) = (( PHI(:,:,hh) * P) * e(:,ii)); 
        elseif method == 'IRFGen'
        % IRF Generalized (Ref.5 p.589)
        IRFGen(:,ii,hh) = ( (PHI(:,:,hh) * SIGMA) * e(:,ii) / sqrt( SIGMA(ii,ii) ) ); 
        else
            error('Hiba')
        end
    end
end

end