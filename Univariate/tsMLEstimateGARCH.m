function [theta,SE,tstat,sig2] = tsMLEstimateGARCH(yt,init,type)
% ------------------------------------------------------------------------------------
% Estimate GARCH(1,1) and GJR-GARCH(1,1,1) models
% ------------------------------------------------------------------------------------
% INPUT: yt: return series (T x 1)
%        init: initial conditions
%        type: GARCH model type (default: GARCH(1,1))
% ------------------------------------------------------------------------------------
% OUTPUT: theta: estimated parameters
%         SE: parameters standard error
%         tstat: t-statistics of the esetimated parameters
%         sig2: fitted sig(t)^2 values
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
    init {mustBeNumeric}
    type char {mustBeMember(type,{'GARCH','GJR'})} = 'GARCH'
end

%----------------------------------(2) set env ---------------------------------------

% Get length of the series
[T,~] = size(yt);

% epsilon(t)
yt = bsxfun(@minus,yt,mean(yt)); % demean return series

% epsilon(t)^2
e2 = yt.^2;

%----------------------------------(3) estimate model --------------------------------

% Estimate model parameters
ops = optimset('LargeScale','off','Display','off');
[theta,~,~,~,~,Hessian] = fminunc(@(theta) tsLLFunctionGARCH(yt,e2,theta,type), init, ops);

% Calculate standard errors
invHess = inv(Hessian);
SE = sqrt(diag(invHess)*1/(T));

% Calculate t-statistics
for i = 1:size(theta,2)
    tstat(i) = theta(i)/SE(i);
end

%----------------------------------(4) fit model -------------------------------------

% Init sigma(t)^2, and set the first value
sig2 = zeros(T,1);
sig2(1) = var(yt);

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

end