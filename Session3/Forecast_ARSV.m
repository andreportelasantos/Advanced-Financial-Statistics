function [h_f, msfe_r,at_1,pt_1] = Forecast_ARSV(par,x,k)
% [h_f,x_f,at_1,pt_1] = Forecast_ARSV(par,x,k)
% par(1) = phi; par(2) = s_eta (also called Q); par(3) = sigma;
% x = returns
% k = prediction horizon
%
% Santiago Pellegrini 2007
%
if size(x,2)>1
    x = x';
end
if nargin<3
    k=10;
end
sigma=par(3);
phi=par(1);
Q=par(2);
[~, ~, at, ~, at_1, pt_1, pt, vt] = KF_ARSV(par(1:2),x);
N = size(x,1);
for t = N+1:N+k
    at_1(t,1) = phi*at(t-1);
    pt_1(t,1) = (phi^2)*pt(t-1) + Q;
    vt(t,1) = 0;
    at(t,1) = at_1(t);
    pt(t,1) = pt_1(t);
    h_f(t-N,1) = at_1(t,1);
    msfe_r(t-N,1) = sigma^2 * exp(h_f(t-N) + 0.5*pt_1(t));
end
% msfe_r(1) = sigma^2 * exp(h_f(1) + 0.5*pt_1(N+1));
% for n=2:k,
%     g(n) = exp(phi.^(n-1)*h_f(1) + 0.5*(phi.^(2*(n-1))*pt_1(N+1) +...
%         sum(phi.^(2*(0:n-2))*Q)));
%     msfe_r(n) = msfe_r(1) + sigma^2 * g(n);
% end



