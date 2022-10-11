function [HARRVdataset, RV_d, RV_w, RV_m] = tsCreateHARdataset(RVdata)
%--------------------------------------------------------------------------
% Create dataset for HAR-RV type volatility model
%
% Calculation of a h-length cascade: RV_t|h = 1/h * [RV(t) + RV(t-1) + ... + RV(t-h)]
%-------------------------------------------------------------------------------
% INPUT: RVdata: volatility series
%-------------------------------------------------------------------------------
% OUTPUT: HARRVdataset: T-by-3 dataset with columns RV_d, RV_w, RV_m
%         RV_d: daily cascade
%         RV_w: weekly cascade
%         RV_m: monthly cascade
%--------------------------------------------------------------------------

if nargin ~= 1
    
    error('Error: RVdata parameter could not be empty')
    
else
    %RVdata=RVdaily;
    wLags = 4;
    mLags = 21;
    nObs = length(RVdata);

    % initiate dataset
    tmpRV_w = zeros(nObs,wLags);
    tmpRV_m = zeros(nObs,mLags);

    % Daily series
    RV_d = RVdata;

    % Create weekly lagged series
    for i = 1:wLags
        tmpRV_w(1+i:nObs,i) = RV_d(1:nObs-i,:);
    end
    preRV_w = [RV_d tmpRV_w];
    
    % Create monthly lagged series
    for i = 1:mLags
        tmpRV_m(1+i:nObs,i) = RV_d(1:nObs-i,:);
    end
    preRV_m = [RV_d tmpRV_m];
    
    % Calculate the cascades
    RV_w = mean(preRV_w,2);
    RV_m = mean(preRV_m,2);
    
    % Merge the cascades to one datamatrix
    HARRVdataset = [RV_d RV_w RV_m];

end

end