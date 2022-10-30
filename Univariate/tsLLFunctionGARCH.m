function LL = tsLLFunctionGARCH(yt,theta,type)
% ------------------------------------------------------------------------------------
% Log-likelihood function to estimate GARCH(1,1) or GJR-GARCH(1,1) models
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
% Version: 1.1    Date: 30/10/2022
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

% Get length of the series
[T,~] = size(yt);
theta = abs(theta);

% Init sig(t)^2, e(t)^2 and leverage vectors
sig2 = zeros(T,1);
e2 = zeros(T,1);
lv2 = zeros(T,1);

%----------------------------------(2) calculate LL function -------------------------

if strcmp(type,'GARCH')
    % Stationary condition check
    gamma = 1 - sum(theta(3:end)); 
    if gamma < 0 
        LL = intmax ; 
        return ; 
    end
    % Set first value of sig(t)^2 and e(t)^2
    sig2(1,:) = theta(2)/gamma; 
    e2(1,:) = (yt(1,:) - theta(1))^2;
    % Start recursive iterate
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1,:) sig2(t-1,:)]*abs(theta(2:end)');
        e2(t,:) = (yt(t,:) - theta(1))^2;
        if sig2(t,:) < 0 
            LL = intmax ; 
            return ; 
        end
    end
elseif strcmp(type,'GJR')
    % Stationary condition check
    gamma = 1 - (sum(theta(3:end-1)) + theta(end)*0.5); 
    if gamma < 0 
        LL = intmax ; 
        return ; 
    end
    % Set first value of sig(t)^2, e(t)^2, and the leverage term
    sig2(1,:) = theta(2)/gamma;
    e2(1,:) = (yt(1,:) - theta(1))^2;
    lv2(1,:) = e2(1,:).*((yt(1,:) - theta(1)) < 0);
    % Start recursive iterate
    for t = 2:T
        sig2(t,:) = [ones(1,1) e2(t-1,:) sig2(t-1,:) lv2(t-1,:)]*abs(theta(2:end)');
        e2(t,:) = (yt(t,:) - theta(1))^2;
        lv2(t,:) = e2(t,:).*( (yt(t,:) - theta(1)) < 0);
        if sig2(t,:) < 0 
            LL = intmax ; 
            return ; 
        end
    end
else
    error('Wrong model type');
end

lt = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(e2./sig2);
LL = -mean(lt);

end