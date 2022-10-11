function [CWstat,pval] = tsCWtest(uhatA,uhatB,yhatA,yhatB,h)
% Clark-West test

% First check whether the two series have the same length
if length(uhatA) == length(uhatB)
    % Get the length of the series
    T = size(uhatA,1);
    
    % Compute losses
    l_A = uhatA.^2;
    l_B = uhatB.^2;
    
    % Compute the loss differential
    d = l_A - l_B + (yhatA - yhatB).^2;

    % Regress the loss differential on a constant
    CWr = nwest(d,ones(T,1),h);

    % Compute the Diebold-Marino test statistic value
    CWstat = CWr.beta/(CWr.se);
    DMst2 = CWr.tstat;
    pval = 1-normcdf(CWstat); 
    %pval = tcdf(-abs(CWstat),T-1);

else
    error('Series length must be the same');
end


end