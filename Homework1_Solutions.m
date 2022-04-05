%%  Problem 1

% Step 1: download SP500 data directly from the Internet 
% You can download the data from the Internet using the data downloader app
% available in:
% https://github.com/Lenskiy/Yahoo-Quandl-Market-Data-Donwloader
% Download the zip file, unzip, and follow the add to path instructions
% similar as in the case of the MFE Toolbox.

data = getMarketDataViaYahoo('^GSPC', '3-Jan-1990', '10-Apr-2020', '1d');
returns = price2ret(data.AdjClose)*100;

% Step 2: descriptive statistics
mean_return=mean(returns);
std_dev=sqrt(var(returns));
kurt=kurtosis(returns);
skew=skewness(returns);
table(mean_return,std_dev,kurt,skew)

% Step 3: autocorrelation of returns up to lag 21
sacf(returns,21);

% Step 4: autocorrelation of squared returns up to lag 21
sacf(returns.^2,21);

% Step 5: cross-correlation of squared returns and lagged returns
lag_returns = returns(1:end-1);
y = returns(2:end); % notice that "lag_returns" and "y" have the same dimension
crosscorr(y.^2,lag_returns,21)


%% Problem 2

% Step 0: estimate an ARMA(1,1) model for the returns and get the residuals
[~, ~, innovations] = armaxfilter(y,1,1,1);    

% Step 1: estimate a GARCH(1,1) model for the series of returns
[parameters_GARCH,LL_GARCH,ht_GARCH] = tarch(innovations,1,0,1);

% Step 2: plots
subplot(3,1,1), plot(innovations), title('Innovations');
subplot(3,1,2), plot(sqrt(ht_GARCH)), title('Conditional standard deviation (volatility)');
subplot(3,1,3), plot(innovations./sqrt(ht_GARCH)), title('Standardized residuals');

% Step 3: Engle and Ng test for the presence of assymetries
% create matrix of regressors
dummy = (lag_returns<0);
standardized_returns = innovations./sqrt(ht_GARCH);
X=[dummy, dummy.*lag_returns, (1-dummy).*lag_returns];
% estimate regressors and check estimated parameters
[B,TSTAT,~,~,~,R2] = ols(standardized_returns.^2,X);
table(B,TSTAT)
% TR2 test
TR2_test = length(standardized_returns)*R2
p_value = 1-chi2cdf(TR2_test,3)





