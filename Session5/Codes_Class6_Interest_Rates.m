%%-------------------------------------------------------
% Estimation of the Diebold and Li (2006) model for the term structure of
% US zero-coupon Treasuries
%------------------------------------------------------

%% Load data

load Data_DieboldLi
maturities = maturities(:);  % ensure a column vector
yields = Data(1:end,:);      % in-sample yields for estimation

% 3D plot of the yields
[X,Y] = meshgrid(dates,maturities);
surf(X,Y,yields')
datetick x
ylabel('maturities (months)')
yticks([3,24,60,96,120]),
zlabel('yields (%)')

%% Estimate the Diebold-Li model using the recommended value for lambda

% Recommended value of lambda 
lambda0 = 0.0609;

% Create factors
level = ones(size(maturities));
slope = (1-exp(-lambda0*maturities))./(lambda0*maturities);
curvature = ((1-exp(-lambda0*maturities))./(lambda0*maturities)-exp(-lambda0*maturities));
X = [level, slope, curvature];

% Create matrix of betas and residuals
beta = zeros(size(yields,1),3);
residuals = zeros(size(yields,1),numel(maturities));

% Run OLS for each period
for i = 1:size(yields,1)
    i
    EstMdlOLS = fitlm(X, yields(i,:)', 'Intercept', false);
    beta(i,:) = EstMdlOLS.Coefficients.Estimate';
    residuals(i,:) = EstMdlOLS.Residuals.Raw';
end

%% Plot yields and betas

subplot(2,1,1),
plot(dates, yields), title('US term structure of zero-coupon Treasuries')
datetick x

subplot(2,1,2),
plot(dates,beta), legend('\beta_0 (level)','\beta_1 (slope)','\beta_2 (curvature)','location','best'), title('Time series of estimated betas')
datetick x

%%  Plot the yield curve in three different points in time

subplot(1,3,1),
y_hat = X*beta(1,:)';
plot(maturities,y_hat)
hold on,
scatter(maturities,yields(1,:),'filled'),
legend('Fitted curve','Observed','location','best'),
ylabel('Yields'), xlabel('Maturities (in months)'),
title(['Yield curve on ',num2str(datestr(dates(1)))])

subplot(1,3,2),
y_hat = X*beta(174,:)';
plot(maturities,y_hat)
hold on,
scatter(maturities,yields(174,:),'filled'),
legend('Fitted curve','Observed','location','best'),
ylabel('Yields'), xlabel('Maturities (in months)'),
title(['Yield curve on ',num2str(datestr(dates(174)))])

subplot(1,3,3),
y_hat = X*beta(end,:)';
plot(maturities,y_hat)
hold on,
scatter(maturities,yields(end,:),'filled'),
legend('Fitted curve','Observed','location','best'),
ylabel('Yields'), xlabel('Maturities (in months)'),
title(['Yield curve on ',num2str(datestr(dates(end)))])

%% Model the betas using a VAR(1)

% Estimate
EstMdlVAR = estimate(varm(3,1), beta);

% View estimated parameters
dispvars = {"Estimated parameters of the VAR(1) model:";...
    "--------------------------------";...
    EstMdlVAR.AR{1}};
cellfun(@disp,dispvars)

%% 12-month ahead forecasts of the yield curve

forecastedBeta = forecast(EstMdlVAR,12,beta);
y_forecast = X*forecastedBeta';

subplot(1,3,1),
plot(maturities,y_forecast(:,1))
ylabel('Yields'), xlabel('Maturities (in months)'),
title('1-month ahead forecasts')

subplot(1,3,2),
plot(maturities,y_forecast(:,6))
ylabel('Yields'), xlabel('Maturities (in months)'),
title('6-month ahead forecasts')

subplot(1,3,3),
plot(maturities,y_forecast(:,12))
ylabel('Yields'), xlabel('Maturities (in months)'),
title('12-month ahead forecasts')


    
