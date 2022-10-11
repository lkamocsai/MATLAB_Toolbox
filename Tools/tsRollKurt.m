function tsRollKurtOut = tsRollKurt(ts,window)
%-------------------------------------------------------------------------------
% Function to calculate rolling kurtosis
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        window: rolling window size
%-------------------------------------------------------------------------------
% OUTPUT: tsRollSkewOut: rolled kurtosis
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

% Get the length of the time series
T = size(ts,1);

% Compute rolling skew
for w = window+1:T
    tmpts = ts(w-window:w-1,:);
    tsRollKurtOut(w-1,:) = kurtosis(tmpts);
end

% Drop the first n rows
tsRollKurtOut = tsRollKurtOut(window:end,:);

end