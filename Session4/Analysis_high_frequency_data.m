%% Load intraday data for Apple stock

load AAPL_data.mat

%% Plot intraday squared returns and volumes for the last available data (16/Dec/2014)

subplot(1,2,1), plot(VOLUMES{end}), title('Traded volume in 16/Dec/2014'),
hold on,
plot(movmean(VOLUMES{end},60)),

subplot(1,2,2), plot(SQ_RETURNS{end}(2:end)), title('Squared 1-min returns in 16/Dec/2014'),
hold on,
plot(movmean(SQ_RETURNS{end}(2:end),60))

%% Compute the standard estimator of the realized variance
% Let's compute the daily realized variance with three alternative sampling
% intervals: 1-minute, 10-minute, 60-minute

for t=1:length(uniqueDates)
    display(['Computing realized variance for date ',num2str(t),' out of ',num2str(length(uniqueDates))])
    % Realized variance with 1-min sampling interval
    fixedInterval = seconds2wall(wall2seconds(93000):60:wall2seconds(160000));
    RV_1M(t) = realized_variance(double(PRICES{t}),double(TIMES{t}),'wall','Fixed',fixedInterval,1);
    % Realized variance with 10-min sampling interval
    fixedInterval = seconds2wall(wall2seconds(93000):600:wall2seconds(160000));
    RV_10M(t) = realized_variance(double(PRICES{t}),double(TIMES{t}),'wall','Fixed',fixedInterval,1);
    % Realized variance with 60-min sampling interval
    fixedInterval = seconds2wall(wall2seconds(93000):3600:wall2seconds(160000));
    RV_60M(t) = realized_variance(double(PRICES{t}),double(TIMES{t}),'wall','Fixed',fixedInterval,1);    
end

%% Plot daily realized variance with 1-min, 10-min, and 60-min sampling frequencies

startDate=datenum('Dec-2003','mmm-yyyy');
endDate=datenum('Dec-2014','mmm-yyyy');
xData = linspace(startDate,endDate,length(uniqueDates))';
xData2 = linspace(startDate,endDate,5);

subplot(4,1,1), 
plot(xData,RETURN_OPEN_TO_CLOSE),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Daily returns');

subplot(4,1,2), 
plot(xData,sqrt(RV_1M)),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Daily realized volatility with 1-minute sampling');

subplot(4,1,3), 
plot(xData,sqrt(RV_10M)),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Daily realized volatility with 10-minute sampling');

subplot(4,1,4), 
plot(xData,sqrt(RV_60M)),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Daily realized volatility with 60-minute sampling');

%% Plot the autocorrelation function of the standardized returns and squared standardized returns

subplot(4,2,1), autocorr(RETURN_OPEN_TO_CLOSE), 
title('returns')

subplot(4,2,2), autocorr((RETURN_OPEN_TO_CLOSE).^2), 
title('sq. returns')

subplot(4,2,3), autocorr(RETURN_OPEN_TO_CLOSE./sqrt(RV_1M)), 
title('standardized returns (1-min RV)')

subplot(4,2,4), autocorr((RETURN_OPEN_TO_CLOSE./sqrt(RV_1M)).^2), 
title('sq. standardized returns (1-min RV)')

subplot(4,2,5), autocorr(RETURN_OPEN_TO_CLOSE./sqrt(RV_10M)), 
title('standardized returns (10-min RV)')

subplot(4,2,6), autocorr((RETURN_OPEN_TO_CLOSE./sqrt(RV_10M)).^2), 
title('sq. standardized returns (10-min RV)')

subplot(4,2,7), autocorr(RETURN_OPEN_TO_CLOSE./sqrt(RV_60M)), 
title('standardized returns (60-min RV)')

subplot(4,2,8), autocorr((RETURN_OPEN_TO_CLOSE./sqrt(RV_60M)).^2), 
title('sq. standardized returns (60-min RV)')


%% Estimate the HAR model using the logs of the 10-min daily realized variance

[parameters, ~, ~, ~, ~, VCV]=heterogeneousar(log(RV_10M'),1,[1 5 10 22]');
t_stat = parameters./sqrt(diag(VCV));
p_values = 1-normcdf(abs(t_stat));
HAR_results = table(parameters,t_stat,p_values)

% extract fitted values
p = [1 1 1 1;1 5 10 22]';
numP = size(p,1);
[Y,X] = newlagmatrix(log(RV_10M)',22,0);
T = length(Y);
newX = zeros(T,numP);
for i=1:numP
    newX(:,i) = mean(X(:,p(i,1):p(i,2)),2);
end
newX = [ones(size(newX,1),1) newX];
RV_10M_hat = newX*parameters;
RV_10M_hat = exp(RV_10M_hat);

%% Compare with the conditional variance from a GARCH(1,1) model 
% estimated for the series of daily open-to-close returns

[parameters_GARCH, ~, ht_GARCH, ~, VCV] = tarch(RETURN_OPEN_TO_CLOSE(23:end)', 1, 0, 1);
t_stat = parameters_GARCH./sqrt(diag(VCV));
p_values = 1-normcdf(abs(t_stat));
GARCH_results = table(parameters_GARCH,t_stat,p_values)

startDate=datenum('Jan-2004','mmm-yyyy');
endDate=datenum('Dec-2014','mmm-yyyy');
xData = linspace(startDate,endDate,length(RETURN_OPEN_TO_CLOSE(23:end)))';
xData2 = linspace(startDate,endDate,5);

subplot(3,1,1), 
plot(xData,RETURN_OPEN_TO_CLOSE(23:end)),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Daily returns');

subplot(3,1,2), 
plot(xData,sqrt(RV_10M_hat)*100), ylim([0,7])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('HAR volatilities');

subplot(3,1,3), 
plot(xData,sqrt(ht_GARCH)), ylim([0,7])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('GARCH volatilities');

%% LM testing

% HAR
[LM_HAR,pval_HAR]=lmtest1((RETURN_OPEN_TO_CLOSE(23:end)./sqrt(RV_10M_hat)').^2,21);
LM_HAR(end)
pval_HAR(end)

% GARCH
[LM_GARCH,pval_GARCH]=lmtest1((RETURN_OPEN_TO_CLOSE(23:end)./sqrt(ht_GARCH)').^2,21);
LM_GARCH(end)
pval_GARCH(end)

 

