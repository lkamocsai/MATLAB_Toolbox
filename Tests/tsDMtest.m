function [DMstat,pval] = tsDMtest(uhatA,uhatB,h)
% Diebold-Marino test

% First check whether the two series have the same length
if length(uhatA) == length(uhatB)
    % Get the length of the series
    T = size(uhatA,1);
    
    % Compute losses
    l_A = uhatA.^2;
    l_B = uhatB.^2;
    
    % Compute the loss differential
    d = l_A - l_B;

    % Regress the loss differential on a constant
    DMr = nwest(d,ones(T,1),h);

    % Compute the Diebold-Marino test statistic value
    DMstat = DMr.beta/(DMr.se);
    DMst2 = DMr.tstat;
    pval = 1-normcdf(DMstat); 
    %pval = tcdf(-abs(DMstat),T-1);

else
    error('The two loss vectors must be the same length');
end


end