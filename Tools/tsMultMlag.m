function tsLagMatOut = tsMultMlag(ts,p)
%-------------------------------------------------------------------------------
% Function to create lag-matrix
%-------------------------------------------------------------------------------
% INPUT: ts: time series (no matter whether the time series uni- or multivariate)
%        p:  number of lags
%-------------------------------------------------------------------------------
% OUTPUT: tsLagMatOut: lagged dataset (note, dataset must be trimmed)
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% https://github.com/lkamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

%--------------------------(1) check inputs, set env----------------------------

arguments
    ts {mustBeNonempty,mustBeNumeric}
    p {mustBeNonempty,mustBeNonzero,mustBeNumeric}
end

% get time series dimensions
[t, K] = size(ts);

% init temp matrix
tmp_lagmat = nan(t,K*p);

% init variables lag position
varPos = 0;

%--------------------------(2) do lagging---------------------------------------

for i = 1:p
    for j = 1:K
        tmp_lagmat(1 + i:t,varPos + j) = ts(1:t - i,j);
    end
    varPos = K * i;
end

%--------------------------(3) save the lagged dataset--------------------------

tsLagMatOut = tmp_lagmat;

end