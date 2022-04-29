%% Solutions for Homework 3

% download data for the SP500
data = getMarketDataViaYahoo('^GSPC', '3-Jan-1990', '10-Apr-2020', '1d');
returns = price2ret(data.AdjClose)*100;

% Fit an ARMA(1,1) model to the data and get the residuals
[parameters_ARMA, LL_ARMA, r] = armaxfilter(returns,1,1,1); 

% fit ARSV model to the residuals of the ARSV model
x0 = [0.93 0.05];
opt = optimset('Display','iter','MaxFunEvals',400,'MaxIter',400,'TolFun',1.0000e-010);
parameters_ARSV = fmincon('KF_ARSV',x0,[],[],[],[],[0 0],[0.99999 inf]',[],opt,r);

% Extract conditional standard deviations of the ARSV model
[~, ~, log_ht] = KF_ARSV(parameters_ARSV,r);
sigma_ARSV = exp(log_ht/2);

% Compare with GARCH(1,1)
[parameters_GARCH,LL_GARCH,ht_GARCH] = tarch(r,1,0,1);
sigma_GARCH = sqrt(ht_GARCH);

% Plots
subplot(3,1,1), plot(sigma_ARSV), title('ARSV conditional standard deviations'), ylim([0;7])
subplot(3,1,2), plot(sigma_GARCH), title('GARCH conditional standard deviations'), ylim([0;7])
subplot(3,1,3), plot(r), title('Residuals of the ARMA(1,1) model')

% LM test for the squared standardized residuals
% LM test for GARCH model
[LM, pval] = lmtest1((r./sigma_GARCH).^2, 21);
LM(end)
pval(end)
% LM test for ARSV model
[LM, pval] = lmtest1((r./sigma_ARSV).^2, 21);
LM(end)
pval(end)

