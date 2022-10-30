function [fitted,LRV] = tsForecastGARCH(fitted,theta,h,type)
% ------------------------------------------------------------------------------------
% Forecast GARCH(1,1) or GJR(1,1)  model h-step ahead
% ------------------------------------------------------------------------------------
% INPUT: yt: return series (T x 1)
%        theta: model parameters
%        h: forecast horizon (default: 1)
% ------------------------------------------------------------------------------------
% OUTPUT: sig2: fitted sigma(t)^2 series with h-step ahead forecast
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
    fitted {mustBeNonempty,mustBeNumeric}
    theta {mustBeNonempty,mustBeNumeric}
    h {mustBeNumeric} = 1
    type char {mustBeMember(type,{'GARCH','GJR'})} = 'GARCH'
end

%----------------------------------(2) set env ---------------------------------------

% Get series length
[TT,~] = size(fitted);

% Estimated params
omega = theta(2);
alpha = theta(3);
beta = theta(4);
if strcmp(type,'GARCH')
    lambda = 0;
else
    lambda = theta(5);
end

%----------------------------------(2) forecast --------------------------------------

% Long run variance
LRV = omega/(1 - alpha - beta - 0.5*lambda); 

% Do the forecast
for hh = 1:h
    fitted(TT + hh,:) = LRV + ((alpha + beta + 0.5*lambda)^(hh - 1))*(fitted(TT,:) - LRV);
end

end