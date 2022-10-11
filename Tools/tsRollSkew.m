function tsRollSkewOut = tsRollSkew(ts,window)
%-------------------------------------------------------------------------------
% Function to calculate rolling skewness
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        window: rolling window size
%-------------------------------------------------------------------------------
% OUTPUT: tsRollSkewOut: rolled skewness
%-------------------------------------------------------------------------------


% Get the length of the time series
T = size(ts,1);

% Compute rolling skew
for w = window+1:T
    tmpts = ts(w-window:w-1,:);
    tsRollSkewOut(w-1,:) = skewness(tmpts);
end

% Drop the first n rows
tsRollSkewOut = tsRollSkewOut(window:end,:);

end