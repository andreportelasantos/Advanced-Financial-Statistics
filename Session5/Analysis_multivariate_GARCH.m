%% Download data

% SPY: S&P 500 ETF Fund
SPY = getMarketDataViaYahoo('SPY', '3-Jan-2010', '01-May-2020', '1d');
SPY = price2ret(SPY.AdjClose)*100;

% IJS: S&P SmallCap 600 Value Idx (ETF)
IJS = getMarketDataViaYahoo('IJS', '4-Jan-2010', '01-May-2020', '1d');
IJS = price2ret(IJS.AdjClose)*100;

% EFA: ETF large- and mid-capitalization developed market equities
EFA = getMarketDataViaYahoo('EFA', '4-Jan-2010', '01-May-2020', '1d');
EFA = price2ret(EFA.AdjClose)*100;

% EEM: SCI Emerging Markets Index (ETF)
EEM = getMarketDataViaYahoo('EEM', '4-Jan-2010', '01-May-2020', '1d');
EEM = price2ret(EEM.AdjClose)*100;

% AGG: iShares Barclays Aggregate Bond Fund
AGG = getMarketDataViaYahoo('AGG', '4-Jan-2010', '01-May-2020', '1d');
AGG = price2ret(AGG.AdjClose)*100;

% Build matrix of returns
returns = [SPY, IJS, EFA, EEM, AGG];

%% Plot returns

startDate=datenum('Jan-2010','mmm-yyyy');
endDate=datenum('May-2020','mmm-yyyy');
xData = linspace(startDate,endDate,length(returns))';
xData2 = linspace(startDate,endDate,5);

subplot(3,2,1), plot(xData,SPY), ylim([-10,10])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('S&P 500 returns');

subplot(3,2,2), plot(xData,IJS), ylim([-10,10])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Small-cap. returns');

subplot(3,2,3), plot(xData,EFA), ylim([-10,10])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Large- and mid-cap. returns');

subplot(3,2,4), plot(xData,EEM), ylim([-10,10])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Emerging markets returns');

subplot(3,2,5), plot(xData,AGG), ylim([-10,10])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Bond returns');

%% Fit RiskMetrics model

Ht_RM = riskmetrics(returns,.94);

%% Fit scalar VECH model

[parameters_VECH, ~, Ht_VECH] = scalar_vt_vech(returns,[],1,0,1);

%% Fit CCC model

[parameters_CCC, ~, Ht_CCC] = ccc_mvgarch(returns,[],1,0,1);

%% Fit DCC model

[parameters_DCC, ~, Ht_DCC] = dcc(returns,[],1,0,1);

%% Plot correlation between SP500 returns and Bond returns

corr_RM=squeeze(Ht_RM(1,5,:))./((squeeze(Ht_RM(1,1,:).^0.5)).*(squeeze(Ht_RM(5,5,:).^0.5)));
corr_VECH=squeeze(Ht_VECH(1,5,:))./((squeeze(Ht_VECH(1,1,:).^0.5)).*(squeeze(Ht_VECH(5,5,:).^0.5)));
corr_CCC=squeeze(Ht_CCC(1,5,:))./((squeeze(Ht_CCC(1,1,:).^0.5)).*(squeeze(Ht_CCC(5,5,:).^0.5)));
corr_DCC=squeeze(Ht_DCC(1,5,:))./((squeeze(Ht_DCC(1,1,:).^0.5)).*(squeeze(Ht_DCC(5,5,:).^0.5)));

startDate=datenum('Jan-2010','mmm-yyyy');
endDate=datenum('May-2020','mmm-yyyy');
xData = linspace(startDate,endDate,length(returns))';
xData2 = linspace(startDate,endDate,5);

subplot(2,2,1), plot(xData,corr_RM), ylim([-1,1])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Stock-bond correlation: RiskMetrics');

subplot(2,2,2), plot(xData,corr_VECH), ylim([-1,1])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Stock-bond correlation: VECH');

subplot(2,2,3), plot(xData,corr_CCC), ylim([-1,1])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Stock-bond correlation: CCC');

subplot(2,2,4), plot(xData,corr_DCC), ylim([-1,1])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('Stock-bond correlation: DCC');

%% Compute (unrestricted) minimum-variance portfolio weights

[t, k]= size(returns); 
e = ones(k,1);
for i=1:(t-1)
    % RM model
    aux = Ht_RM(:,:,i)\e;
    weights_RM(:,i) = aux/sum(aux);
    portfret_RM(i)  = returns(i+1,:)*weights_RM(:,i);
    % VECH model
    aux = Ht_VECH(:,:,i)\e;
    weights_VECH(:,i) = aux/sum(aux);
    portfret_VECH(i)  = returns(i+1,:)*weights_VECH(:,i);
    % CCC model
    aux = Ht_CCC(:,:,i)\e;
    weights_CCC(:,i) = aux/sum(aux);
    portfret_CCC(i)  = returns(i+1,:)*weights_CCC(:,i);
    % DCC model
    aux = Ht_DCC(:,:,i)\e;
    weights_DCC(:,i) = aux/sum(aux);
    portfret_DCC(i)  = returns(i+1,:)*weights_DCC(:,i);
end

%% Plot unconstrained portfolio weights 

startDate=datenum('Jan-2010','mmm-yyyy');
endDate=datenum('May-2020','mmm-yyyy');
xData = linspace(startDate,endDate,length(returns)-1)';
xData2 = linspace(startDate,endDate,5);

subplot(2,2,1), plot(xData,weights_RM), ylim([-1,1.5])
legend('SPY','IJS','EFA','EEM','AGG'),
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('RiskMetrics weights');

subplot(2,2,2), plot(xData,weights_VECH), ylim([-1,1.5])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('VECH weights');

subplot(2,2,3), plot(xData,weights_CCC), ylim([-1,1.5])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('CCC weights');

subplot(2,2,4), plot(xData,weights_DCC), ylim([-1,1.5])
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('DCC weights');

%% Plot cumulative portfolio returns of unconstrained portfolios

startDate=datenum('Jan-2010','mmm-yyyy');
endDate=datenum('May-2020','mmm-yyyy');
xData = linspace(startDate,endDate,length(returns)-1)';
xData2 = linspace(startDate,endDate,5);

subplot(2,2,1), plot(xData,cumprod(1+portfret_RM/100)-1), 
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('RM portf. returns');

subplot(2,2,2), plot(xData,cumprod(1+portfret_VECH/100)-1), 
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('VECH portf. returns');

subplot(2,2,3), plot(xData,cumprod(1+portfret_CCC/100)-1), 
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('CCC portf. returns');

subplot(2,2,4), plot(xData,cumprod(1+portfret_DCC/100)-1), 
set(gca,'XTick',xData2);
datetick('x','mmmyy','keepticks');
title('DCC portf. returns');

%% Compute shortsell-constrained portfolios

% Function to compute portfolio variance (objective function)
fun = @(w,H) w'*H*w;

% Inputs to the contrained optimization via fmincon function
opt = optimset('Display','off');
w0 = ones(size(returns,2),1)/size(returns,2); % initial condition
Aeq = ones(1,size(returns,2)); % Equality constraint
beq = 1; % weights sum up to 1
lb = zeros(1,size(returns,2)); % lower bound of 0
ub = ones(1,size(returns,2)); % upper bound of 1

% Obtain constrained portfolios for the case of the CCC model
for i=1:(t-1)
    display(i)
    % CCC model
    weights_CCC_constrained(:,i) = fmincon(fun,w0,[],[],Aeq,beq,lb,ub,[],opt,Ht_CCC(:,:,i));
    portfret_CCC_constraine(i)  = returns(i+1,:)*weights_CCC_constrained(:,i);
end





