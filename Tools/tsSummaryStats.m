function tsbstsOut = tsSummaryStats(ts,varargin)
%-------------------------------------------------------------------------------
% Calculate the conventional time series statistics (mean, standard deviation, 
% skewness, kurtosis, Jarque-Bera test, Ljung-Box test, ARCH LM test and ADF test)
%-------------------------------------------------------------------------------
% INPUT: ts: time series
%        alpha (optional): sig level (default: 5%)
%-------------------------------------------------------------------------------
% OUTPUT: tsbstsOut: print basic statistics
%-------------------------------------------------------------------------------


%Base statistics(mean, std, skew, kurt, ACF,JB,LB)
[~,c] = size(ts);
for i = 1:c
% 1/a Mean (0 kornyeki erteknek kell lennie)
    tsbstsOut(i,1) = mean(ts(:,i));
% 1/b Standard deviation (ha standardizalt az adat akkor 1 koruli ertek, amugy barmi)
    tsbstsOut(i,2) = std(ts(:,i));
% 1/c Skewness (normal esetben = 0, ha skew > 0 jobbra van nagyobb tomeg, ha <0 akkor balra)
    tsbstsOut(i,3) = skewness(ts(:,i));
% 1/d Kurtosis (normal esetben = 3, ha nagyobb mint 3 vastagfarku az eloszlas)
    tsbstsOut(i,4) = kurtosis(ts(:,i));
% 1/e Normality ,Jarque-Bera test (ha pval > siglevel akkor normalis az eloszlas)
    [JBval,tsbstsOut(i,5),H] = tsJBtest(ts(:,i),varargin{:});
% 1/f.1 k-lag autocorrelation, Ljung-Box test (ha pval > siglevel akkor no autocorrelation)
    [Q1val,tsbstsOut(i,6),H] = tsLBQtest(ts(:,i),1,0.05);
% 1/f.2 k-lag autocorrelation, Jung-Box test (ha pval > siglevel akkor no autocorrelation)
    [Q5val,tsbstsOut(i,7),H] = tsLBQtest(ts(:,i),5,0.05);
% 1/f.3 k-lag autocorrelation, Jung-Box test (ha pval > siglevel akkor no autocorrelation)
    [Q22val,tsbstsOut(i,8),H] = tsLBQtest(ts(:,i),22,0.05);
% 1/g Heteroscedasticity, ARCH LM test (ha pval > siglevel akkor homoscedasticity)
    [ARCHLMval,tsbstsOut(i,9),H] = tsARCHLMtest(ts(:,i),5,0.05);
% 1/h Stacionaritas, ADF test (ha pval < siglevel akkor stacioner)
    [~,tsbstsOut(i,10),~,~,~] = adftest((ts(:,i)),'alpha',0.05);

end

% %--------------------(2) print results %----------------------------------------
% disp(['Mean:        ', sprintf('%.4f',  tsbstsOut(i,1))]);
% disp(['Std:         ', sprintf('%.4f',  tsbstsOut(i,2))]);
% disp(['Skewness:    ', sprintf('%.4f',  tsbstsOut(i,3))]);
% disp(['Kurtosis:    ', sprintf('%.4f',  tsbstsOut(i,4))]);
% disp(['JB test:     ', sprintf('%.2f',  tsbstsOut(i,5))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,6))]);
% disp(['Q(1):        ', sprintf('%.2f',  tsbstsOut(i,7))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,8))]);
% disp(['Q(5):        ', sprintf('%.2f',  tsbstsOut(i,9))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,10))]);
% disp(['Q(22):       ', sprintf('%.2f',  tsbstsOut(i,11))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,12))]);
% disp(['ARCH LM:     ', sprintf('%.2f',  tsbstsOut(i,13))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,14))]);
% disp(['ADF:         ', sprintf('%.2f',  tsbstsOut(i,15))]);
% disp(['             ', sprintf('(%.4f)',tsbstsOut(i,16))]);

end