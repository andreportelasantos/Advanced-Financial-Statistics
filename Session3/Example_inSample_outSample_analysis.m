% Get data
data = getMarketDataViaYahoo('^GSPC', '3-Jan-1990', '10-Apr-2020', '1d');
returns = price2ret(data.AdjClose)*100;

% Split data
inSampleData = returns(1:floor(length(returns)/2));
outSampleData = returns(floor(length(returns)/2)+1:end);

% Fit ARMA(1,1) for the in-sample data
Mdl = arima(1,0,1);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,inSampleData);

% Get ARMA residuals for the in-sample data
inSampleResid = infer(EstMdl,inSampleData);

% Get ARMA residuals for the out-sample data
outSampleResid = infer(EstMdl,outSampleData);

% Fit ARSV model for the in-sample data 
x0 = [0.93 0.05];
opt = optimset('Display','iter','MaxFunEvals',400,'MaxIter',400,'TolFun',1.0000e-010);
[parameters_ARSV] = fmincon('KF_ARSV',x0,[],[],[],[],[0 0],[0.99999 inf]',[],opt,inSampleResid);

% Get in-sample variances
[~, ~, log_ht] = KF_ARSV(parameters_ARSV,inSampleResid);
sigma_ARSV_inSample = exp(log_ht/2);

% Get out-sample variances
[~, ~, log_ht] = KF_ARSV(parameters_ARSV,outSampleResid);
sigma_ARSV_outSample = exp(log_ht/2);

% plot in-sample and out-sample conditional variances
plot(sigma_ARSV_inSample)
hold on
plot(length(inSampleData)+1:length(returns),sigma_ARSV_outSample,'r'), legend('in-sample','out-sample'),




