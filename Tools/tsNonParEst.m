function [f,grid] = tsNonParEst(ts,grid)
%-------------------------------------------------------------------------------
% Function to non-parametric estimate of a distribution
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        k: lag number (default: 22)
%-------------------------------------------------------------------------------
% OUTPUT: f:
%	  grid:
%-------------------------------------------------------------------------------

%----------------------(1) input check, base calculations-----------------------

min_ts = min(ts);
max_ts = max(ts);

switch nargin
    case 1
        grid = (min_ts:0.001:max_ts)';
    case 2
    otherwise
        error('minimum input required: time series vector')
end

ord_ts = sort(ts);
std_ts = std(ts);
T = length(ts);

%-----------------------(2) calculate optimal bandwidth-------------------------

h = 1.06 * std_ts * T^(-1/5);

%---------------------------(3) kernel estimate---------------------------------

f = zeros(length(grid),1);
for i = 1:length(grid)
    f(i) = mean((1/h) * (1/sqrt(2*pi)) * exp( (-1/2) * ((grid(i)-ord_ts)./h).^2 ) );
end

end