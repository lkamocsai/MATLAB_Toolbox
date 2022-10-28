function [FEVDOrth,FEVDGen,MSE] = tsFEVD(PHI,SIGMA,IRFOrth,IRFGen,h)
% ------------------------------------------------------------------------------------
% Function to calculate Forecast Error Variance Decomposition
% ------------------------------------------------------------------------------------
% INPUT: PHI: MA weight matrices (K x K x h)
%        SIGMA: Covariance matrix (K x K)
%        IRFOrth: Orthogonalized Inpulse Response (K x K x h)
%        IRFGen: Generalized Inpulse Response (K x K x h)
%        h: number of steps ahead 
% ------------------------------------------------------------------------------------
% OUTPUT: FEVDOrth: Orthogonalized Forecast Error Variance Decomposition (K x K x h)
%         FEVDGen: Generalized Forecast Error Variance Decomposition (K x K x h)
%         MSE: Mean Square Errors (K x K x h)
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
    PHI {mustBeNonempty,mustBeNumeric}
    SIGMA {mustBeNonempty,mustBeNumeric}
    IRFOrth {mustBeNonempty,mustBeNumeric}
    IRFGen {mustBeNonempty,mustBeNumeric}
    h {mustBeNonempty,mustBeNumeric}
end

% ------------------------------(2) set env ------------------------------------------

K = size(IRFOrth,1);
e = eye(K); % selection vector
MSE = zeros(K,1,h);
tmpMSE = zeros(K,1);
tmpIRFOrth = zeros(K,K);
tmpIRFGen = zeros(K,K);

% ------------------------------(3) Orthogonalizes and generalized FEVD --------------

for hh = 1:h
    for ii = 1:K
        % Mean Square Error
        MSE(ii,:,hh) = e(:,ii)' * ( PHI(:,:,hh) * SIGMA * PHI(:,:,hh)' ) * e(:,ii);
    end
    % h-step cumulative sum of the variable k innnovation contribution to the FEV
    csumsqIRFOrth(:,:,hh) = tmpIRFOrth + IRFOrth(:,:,hh).^2;
    tmpIRFOrth = csumsqIRFOrth(:,:,hh);
    csumsqIRFGen(:,:,hh) = tmpIRFGen + IRFGen(:,:,hh).^2;
    tmpIRFGen = csumsqIRFGen(:,:,hh);
    % cumulative sum of variables over forecast horizons per equations (row)
    MSE(:,:,hh) = tmpMSE + MSE(:,:,hh); 
    tmpMSE = MSE(:,:,hh);
end

% Othogonalized and generalized forecast error variance decomposition (Ref.5 p.592)
FEVDOrth = (csumsqIRFOrth./MSE)*100;
FEVDGen = (csumsqIRFGen./MSE)*100;

end