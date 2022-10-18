function LL = tsMLEstimateGARCH(yt,theta,type)
% ------------------------------------------------------------------------------------
% Log-likelihood function to estimate GARCH(1,1), GJR-GARCH models
% ------------------------------------------------------------------------------------
% INPUT: yt: return series (T x 1)
%        theta: model parameters
%        type: GARCH model type (default: GARCH(1,1))
% ------------------------------------------------------------------------------------
% OUTPUT: LL: log-likelihood (scalar)        
% ------------------------------------------------------------------------------------
% Refrences:
% 1. H.Lütkepohl - New Introduction to Multiple Time Series Analysis (Springer, 2005)
% 2. S.Hurn, V.Martin, D.Harris - Econometric Modelling with Time Series (Cambridge, 2012)
% 3. J.D.Hamilton - Time Series Analysis (Princeton, 1994)
% ------------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% https://github.com/lkamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------------
%
%----------------------------------(1) check inputs ----------------------------------

arguments
    yt {mustBeNumeric}
    theta {mustBeNumeric}
    type char {mustBeMember(type,{'GARCH','GJR'})} = 'GARCH'
end

%----------------------------------(2) set env ---------------------------------------

% Get series length
[T,~] = size(yt);

% Demean return series, epsilon(t)
yt = bsxfun(@minus,yt,mean(yt)); 

% epsilon(t)^2
e2 = yt.^2;

% Init sigma(t)^2, and set the first value
sig2 = zeros(T,1);
sig2(1) = var(yt); 

%----------------------------------(2) calculate LL function -------------------------

if strcmp(type,'GARCH')
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1) sig2(t-1)]*theta';
    end
elseif strcmp(type,'GJR')
    % Get the levarage term
    lv2 = e2.*(yt < 0);
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1) sig2(t-1) lv2(t-1)]*theta';
    end
else
    error('Wrong model type');
end

lt = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(e2./sig2);
LL = -mean(lt);

end