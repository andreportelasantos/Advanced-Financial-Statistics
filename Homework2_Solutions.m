%%  Problem 1

% Step 1: download SP500 data directly from the Internet 
% You can download the data from the Internet using the data downloader app
% available in:
% https://github.com/Lenskiy/Yahoo-Quandl-Market-Data-Donwloader
% Download the zip file, unzip, and follow the add to path instructions
% similar as in the case of the MFE Toolbox.

data = getMarketDataViaYahoo('^GSPC', '3-Jan-1990', '10-Apr-2020', '1d');
returns = price2ret(data.AdjClose)*100;

% Step 1: estimate an ARMA(1,1) model for the returns and get the residuals
[~, ~, innovations] = armaxfilter(returns,1,1,1);    

% Step 2: estimate a GJR model for the series of residuals
[parameters_GJR,LL_GJR,ht_GJR,VCV_GJR] = tarch(innovations,1,1,1);

% Step 3: plots
subplot(3,1,1), plot(innovations), title('Innovations');
subplot(3,1,2), plot(sqrt(ht_GJR)), title('Conditional standard deviation (volatility) from GJR model');
subplot(3,1,3), plot(innovations./sqrt(ht_GJR)), title('Standardized residuals');

% Step 4: estimate a GARCH model for the series of residuals
[parameters_GARCH,LL_GARCH,ht_GARCH,VCV_GARCH] = tarch(innovations,1,0,1);

% Step 5: compare volatilities of the GJR and GARCH models
subplot(2,1,1), plot(sqrt(ht_GJR)), title('Conditional standard deviation (volatility) from GJR model');
subplot(2,1,2), plot(sqrt(ht_GARCH)), title('Conditional standard deviation (volatility) from GARCH model');

% Step 6: compare the t-statistics of the estimated parameters of both
% models
% GJR model
parameters = {'omega','alpha','lambda','beta'}';
t_stat_GJR = parameters_GJR./sqrt(diag(VCV_GJR));
pval_GJR = 1 - normcdf(abs(t_stat_GJR));
table(parameters,parameters_GJR,t_stat_GJR,pval_GJR)
% GARCH model
parameters = {'omega','alpha','beta'}';
t_stat_GARCH = parameters_GARCH./sqrt(diag(VCV_GARCH));
pval_GARCH = 1 - normcdf(abs(t_stat_GARCH));
table(parameters,parameters_GARCH,t_stat_GARCH,pval_GARCH)

% Step 7: tests
% LR test
LR_test = 2*(LL_GJR-LL_GARCH)
pval_LR_test = 1-chi2cdf(LR_test,1)

% AIC and BIC
T = length(innovations);
[aic,bic] = aicbic([LL_GARCH;LL_GJR], [3; 4], T*ones(2,1))

% LM test for GARCH model
[LM, pval] = lmtest1((innovations./sqrt(ht_GARCH)).^2, 21);
LM(end)
pval(end)
% LM test for GJR model
[LM, pval] = lmtest1((innovations./sqrt(ht_GJR)).^2, 21);
LM(end)
pval(end)

