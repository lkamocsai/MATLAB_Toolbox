function [theta,SE,tstat,sig2] = tsMLEstimateGARCH(yt,init,type)
% ------------------------------------------------------------------------------------
% Estimate GARCH(1,1) or GJR-GARCH(1,1) models
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
% Version: 1.1    Date: 30/10/2022
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
% Calculate log-returns
yt = price2ret(yt, Method="continuous");

% Get length of the series
[T,~] = size(yt);

%----------------------------------(3) estimate model --------------------------------

% Estimate model parameters
ops = optimset('LargeScale','off','Display','off');
[theta,~,~,~,~,Hessian] = fminunc(@(theta) tsLLFunctionGARCH(yt,theta,type), init, ops);

% Calculate standard errors using the Hessian
invHess = inv(Hessian);
SE = sqrt(abs(diag(invHess))*(1/T));

% Calculate coefficients t-statistics
for i = 1:size(theta,2)
    tstat(i) = theta(i)/SE(i);
end

%----------------------------------(4) fit model -------------------------------------

% Init sig(t)^2, and set the first value
sig2 = zeros(T,1);
e2 = zeros(T,1);
lv2 = zeros(T,1);

if strcmp(type,'GARCH') 
    gamma = 1 - sum(theta(3:end)); 
    sig2(1,:) = theta(2)/gamma; 
    e2(1,:) = (yt(1,:) - theta(1))^2;
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1,:) sig2(t-1,:)]*abs(theta(2:end)'); 
        e2(t,:) = (yt(t,:) - theta(1))^2;
    end
elseif strcmp(type,'GJR')
    gamma = 1 - (sum(theta(3:end-1)) + theta(end)*0.5); 
    sig2(1,:) = theta(2)/gamma;
    e2(1,:) = (yt(1,:) - theta(1))^2;
    lv2(1,:) = e2(1,:).*((yt(1,:) - theta(1)) < 0);
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1,:) sig2(t-1,:) lv2(t-1,:)]*abs(theta(2:end)');
        e2(t,:) = (yt(t,:) - theta(1))^2;
        lv2(t,:) = e2(t,:).*( (yt(t,:) - theta(1)) < 0);
    end
else
    error('Wrong model type');
end

end