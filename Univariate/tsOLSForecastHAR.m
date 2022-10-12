function modelfit = tsOLSForecastHAR(HARData,h,window)
%-------------------------------------------------------------------------------
% In-Sample Maximum Likelihood estimation for HAR models
%
%-------------------------------------------------------------------------------
% INPUT: HARData: HAR time series
%        h: estimation horizon
%        window: windows size of forecast
%-------------------------------------------------------------------------------
% OUTPUT: yhat: estimated values
%         bhat: estimated parameters
%         uhat: residuals
%         MSE: MSE loss function
%         HRMSE: Heteroscedasticity Robust MSE loss function
%         QLIKE: Q-Like loss function
%         MZ: Mincer-Zarnowiz regression
% 
%-------------------------------------------------------------------------------
% References:
% 1. Fulvio Corsi, A Simple Approximate Long-Memory Model of Realized Volatility, 
%    Journal of Financial Econometrics, Volume 7, Issue 2, Spring 2009, Pages 174–196, 
%    https://doi.org/10.1093/jjfinec/nbp001
% 2. Corsi, F., & Renò, R. (2012). Discrete-Time Volatility Forecasting With Persistent Leverage Effect 
%    and the Link With Continuous-Time Volatility Modeling. Journal of Business & Economic Statistics, 
%    30(3), 368–380. https://doi.org/10.1080/07350015.2012.663261 
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------
%
%--------------------(1) input check, base calculations ------------------------

%set which column to use as dependent variable (y)
if h == 1 || h == 5 || h == 22 
    switch h
        case 1
            c = 1;
        case 5
            c = 2;
        case 22
            c = 3; 
    end
end 

% Get the series length
[T,~] = size(HARData);

%--------------------(2) estimate model parameters ----------------------------

% Out-of-sample forecast using rolling window
% Pre-allocate yhatrw
yhatrw = zeros(T,2);

for w = window+1:T
    %Set windowed HAR data
    tmpX = [ones(window,1) HARData(w-window:w-1,:)];
    tmpy = HARData(w-window:w-1,c);
    X = tmpX(22:end,:);
    y = tmpy(22:end,:);

    % Estimate model by OLS
    bhatrw(w,:) = (X(1:end-h,:)'*X(1:end-h,:))\(X(1:end-h,:)'*y(1+h:end));

    % Save the actual and estimated RV value
    yhatrw(w,1) = HARData(w,c);
    yhatrw(w,2) = X(end-h+1,:)*bhatrw(w,:)';
    %tmpuhatrw(:,w) = X*bhatrw(w,:)' - y;
end 
    
% Out-of-sample forecast using expanding window
% Pre-allocate yhatiw
yhatiw = zeros(T,2);
    
for w = window+1:T
    %Set windowed HAR data
    tmpX = [ones(w-1,1) HARData(1:w-1,:)];
    tmpy = HARData(1:w-1,c);
    X = tmpX(22:end,:);
    y = tmpy(22:end,:);          

    % Estimate model by OLS
    bhatiw(w,:) = (X(1:end-h,:)'*X(1:end-h,:))\(X(1:end-h,:)'*y(1+h:end));

    % Save the actual and the estimated RV value
    yhatiw(w,1) = HARData(w,c);
    yhatiw(w,2) = X(end-h+1,:)*bhatiw(w,:)';
    %tmpuhatiw(:,w) = X*bhatiw(w,:)' - y;
end

%--------------------(3) run test and save results -----------------------------

% Save the model parameters and the estimated y values
yhat(:,:,1) = yhatrw; 
yhat(:,:,2) = yhatiw;
bhat(:,:,1) = bhatrw; 
bhat(:,:,2) = bhatiw;

% Drop the first n (= window) rows
modelfit.yhat = yhat(window+1:end,:,:);
modelfit.bhat = bhat(window+1:end,:,:);
%modelfit.uhatrw = tmpuhatrw(:,window+1:end);
%modelfit.uhatiw = tmpuhatiw(:,window+1:end);

% Save the residuals and compute the loss functions
yhatsize = size(modelfit.yhat);

for i = 1:yhatsize(3)
    modelfit.uhat(:,i) = modelfit.yhat(:,1,i) - modelfit.yhat(:,2,i);
    modelfit.MSE(i) = sqrt(mean((modelfit.yhat(:,1,i) - modelfit.yhat(:,2,i)).^2));
    modelfit.HMSE(i) = mean( ( ( modelfit.yhat(:,2,i)./modelfit.yhat(:,1,i) ) -1 ).^2 );
    modelfit.QLIKE(i) = mean(log(modelfit.yhat(:,1,i)) + (modelfit.yhat(:,2,i)./modelfit.yhat(:,1,i)));
end

% Forecast evaluation using MZ regression
for i = 1:yhatsize(3)
    modelfit.MZ(i) = nwest(modelfit.yhat(:,1,i),[ones(yhatsize(1),1) modelfit.yhat(:,2,i)]);
end

end