function tLL = tsLLstudent(y,X,theta)
%-------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------

[~,K] = size(X);

beta = theta(1:K);
sigma = abs(theta(K + 1));
v = abs(theta(K + 2));
u = y - X * beta';
z = u./sigma;

% log-likelihood function
const = gamma((v + 1)/2)/(sqrt(pi*(v - 2))*gamma(v/2));
tLL = log(const) - 0.5*log(sigma^2) - ( (v + 1)/2 ) * log(1 + (z.^2)/((v - 2))) ;

tLL = -mean(tLL);

end