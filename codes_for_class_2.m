%%  Model selection and testing: example 1
% Using the MFE_Toolbox
% Load data
load('ibm.mat') % you can use your own time series
% Fit GARCH(1,1)
[parameters_GARCH,LL_GARCH,Ht_GARCH] = tarch(r*100,1,0,1);
% Fit GJR-GARCH(1,1)
[parameters_GJR,LL_GJR,Ht_GJR] = tarch(r*100,1,1,1);
% LR test
LR_test = 2*(LL_GJR-LL_GARCH)
% p-value of the LR test

%%   Model selection and testing: example 2
% Compute AIC and BIC for the previous example
% (using the function of the Econometrics Toolbox)
T = length(r);
[aic,bic] = aicbic([LL_GARCH;LL_GJR], [3; 4], T*ones(2,1))

%% Residual-based model diagnosis

% GARCH(1,1) model
[LM, pval] = lmtest1((r./sqrt(Ht_GARCH)).^2, 21);
LM(end)
pval(end)

% GJR-GARCH(1,1) model
[LM, pval] = lmtest1((r./sqrt(Ht_GJR)).^2, 21);
LM(end)
pval(end)

%% Volatility forecasting
Mdl = garch('Constant',0.02,'GARCH',0.85,'ARCH',0.05);
rng default; % For reproducibility
[v,y] = simulate(Mdl,100);
vF1 = forecast(Mdl,30,’Y0’,y);
% plot
figure
plot(v,'Color',[.7,.7,.7])
hold on
plot(101:130,vF1,'r','LineWidth',2)
title('Forecasted Conditional Variances')
legend('Observed','Forecasts','Location','NorthEast')
hold off

