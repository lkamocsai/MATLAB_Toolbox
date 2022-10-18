function sig2 = tsForecastGARCH(yt,theta,h)
% ------------------------------------------------------------------------------------
% Fit GJR-GARCH model and forecast h-step ahead
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
    yt {mustBeNonempty,mustBeNumeric}
    theta {mustBeNonempty,mustBeNumeric}
    h {mustBeNumeric} = 1
end

%----------------------------------(2) set env ---------------------------------------

% Get series length
[T,~] = size(yt);

% epsilon(t)
yt = bsxfun(@minus,yt,mean(yt)); % demean return series

% epsilon(t)^2
e2 = yt.^2;

% Init sigma(t)^2, and set the first value
sig2 = zeros(T,1);
sig2(1) = var(yt);

% Get the levarage term
lv2 = e2.*(yt < 0); 

%----------------------------------(2) fit model -------------------------------------

for t = 2:T
    sig2(t,:) = [ones(1,1) e2(t-1) sig2(t-1) lv2(t-1)]*theta'; 
end

%----------------------------------(2) forecast --------------------------------------

% Get sigma(t)^2 length
[TT,~] = size(sig2);

% Estimated params
omega = theta(1);
alpha = theta(2);
beta = theta(3);
lambda = theta(4);

% Long run variance
LRV = omega/(1 - alpha - beta - 0.5*lambda); 

% Do the forecast
for hh = 1:h
    sig2(TT + hh,:) = LRV + ((alpha + beta + 0.5*lambda)^(hh - 1))*(sig2(TT,:) - LRV);
end

end