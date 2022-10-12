function modelfit = tsMLEstimateHAR(HARData,h,distribution,alpha)
%-------------------------------------------------------------------------------
% In-Sample Maximum Likelihood estimation for HAR models
%
%-------------------------------------------------------------------------------
% INPUT: HARData: HAR time series
%        h: estimation horizon
%        distribution: dist. assumed to estimate model parameters
%        alpha: sig level for test the null hyp. H0: theta(i) = 0
%-------------------------------------------------------------------------------
% OUTPUT: h: horizon
%         X: independent variables
%         y: dependent variable
%         theta1: estimated parameters
%         LogL1: log-likelihood
%         Gt1: Gradient vector
%         HT1: Hessian Matrix
%         yhat: estimated values
%         uhat: residuals
%         MZ: Mincer-Zarnowiz regression
%         R2: Coefficient of determination
%         MSE: MSE loss function
%         HRMSE: Heteroscedasticity Robust MSE loss function
%         QLIKE: Q-Like loss function
%         stdE: Standard Error
%         NWVhatT: Newey-West robust VCV
%         NWRobustSE: Newey-West robust SE
%         W: Wald test
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

[T,~] = size(HARData);
modelfit.h = h;
modelfit.X = [ones(T-22-h+1,1) HARData(22:end-h,:)];
modelfit.y = HARData(22+h:end,c); 
[t,K] = size(modelfit.X);

%--------------------(2) estimate model parameters ----------------------------

% Set optimization parameters
ops = optimset('LargeScale','off','Display','off');

% Select distribution to use to estimate parameters
switch distribution
    case 'normal'
        % Set initial values for theta
        inittheta = ones(1,K+1);

        % Estimate the unrestricted model
        [modelfit.theta1,modelfit.logL1,~,~,modelfit.Gt1,modelfit.Ht1] = fminunc(@(theta) tsLLnormal(modelfit.y,modelfit.X,theta), inittheta, ops);
        
        % Compute the fitted values
        modelfit.yhat = modelfit.X * modelfit.theta1(1:end-1)';

    case 'student'
        % Set initial values for theta
        inittheta = ones(1,K+2);
        inittheta(K+2) = 8;

        % Estimate the unrestricted model
        [modelfit.theta1,modelfit.logL1,~,~,modelfit.Gt1,modelfit.Ht1] = fminunc(@(theta) tsLLstudent(modelfit.y,modelfit.X,theta), inittheta, ops);

        % Compute the fitted values
        modelfit.yhat = modelfit.X*modelfit.theta1(1:end-2)';
end

%--------------------(3) run test and save results -----------------------------

% Compute residuals
modelfit.uhat = modelfit.y - modelfit.yhat;

% Evaluate estimate using MZ regression
modelfit.MZ = nwest(modelfit.y,[ones(size(modelfit.yhat,1),1) modelfit.yhat]);
modelfit.MZ.chi2stat = (modelfit.MZ.beta-[0;1])'*inv(modelfit.MZ.vcv)*(modelfit.MZ.beta-[0;1]);
modelfit.R2 = 1-cov(modelfit.y - modelfit.yhat)/cov(modelfit.y);

% Calculate loss functions
modelfit.MSE = sqrt(mean((modelfit.y - modelfit.yhat).^2));
modelfit.HRMSE = sqrt(mean( ((modelfit.y - modelfit.yhat)./modelfit.y).^2 ));
modelfit.QLIKE = mean(log(modelfit.y) + (modelfit.yhat./modelfit.y));

% Calculate standard errors using the Hessian
modelfit.stdE = tsMLEstdErr(modelfit.Ht1,t);

% Calculate NW robust standard errors
if h == 1
    [modelfit.NWVhatT,modelfit.NWRobustSE] = tsNWRobustSE(modelfit.X,modelfit.uhat,h);
    modelfit.nwest = nwest(modelfit.y,modelfit.X,h);
else
    [modelfit.NWVhatT,modelfit.NWRobustSE] = tsNWRobustSE(modelfit.X,modelfit.uhat,h*2);
    modelfit.nwest = nwest(modelfit.y,modelfit.X,h*2);
end

% Single parameter Wald test which in this case equal with a simple t-test
vtheta1 = (diag(modelfit.NWVhatT));
dof = t-1;
for i = 1:K
    modelfit.W.val1(i) = (modelfit.theta1(i)./sqrt(vtheta1(i))); %t-stat formula from K. Sheppard *sqrt(t)
    modelfit.W.pv1(i) = (tcdf(-abs(modelfit.W.val1(i)),dof));
    %modelfit.W.pv1(i) = 1 - (chi2cdf(modelfit.W.val1(i),dof));
    modelfit.W.H1(i) = modelfit.W.pv1(i) <= alpha; 
end

end