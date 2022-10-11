function tsklagacorr = tsACF(ts,k)
%-------------------------------------------------------------------------------
% Calculate k-lag autocorrelation
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        k: lag number (default: 5)
%-------------------------------------------------------------------------------
% OUTPUT: tsklagacorr: k-lag autocorrelation
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

%----------------------------(1) input check------------------------------------
switch nargin
    case 1
        k = 5;
    case 2
    otherwise
        error('minimum input required: time series vector')
end

t = length(ts);

%----------------------------(2) calculate ACF----------------------------------
for i = 1:k+1
    y = ts(1+i-1:t,:);
    y_lagged = [ones(t-i+1,1) ts(1:t-i+1,:)];
    tsklagacorr(i,:) = regress(y,y_lagged);
end

tsklagacorr = tsklagacorr(:,2);
end