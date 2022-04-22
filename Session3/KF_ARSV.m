function [LogV, Gr, at, Ft, at_1, pt_1, pt, vt] = KF_ARSV(par,x)
% [LogV Gr at Ft at_1 pt_1 pt vt] = KF_ARSV(par,x)

if size(x,2)>1
    x = x';
end
z = log(x.^2);
%gama = mean(z);
%y = z - gama;
y = z + 1.27;
phi = par(1);
Q = par(2);
H = (pi^2)/2;

N = size(y,1);
ep = 0.0001;

at(1,1) = 0;
pt(1,1) = Q;
LogV = 0;

phip = phi + ep;
atp(1,1) = 0;
ptp(1,1) = Q;
LogVp = 0;

Qq = Q + ep;
atq(1,1) = 0;
ptq(1,1) = Qq;
LogVq = 0;

for t = 2:N
    at_1(t,1) = phi*at(t-1);
    pt_1(t,1) = (phi^2)*pt(t-1) + Q;
    Ft(t,1) = pt_1(t,1) + H;
    vt(t,1) = y(t) - at_1(t,1);
    at(t,1) = at_1(t) + pt_1(t).*(1./Ft(t)).*vt(t);
    pt(t,1) = pt_1(t) - (pt_1(t).^2).*(1./Ft(t));
    LogV = LogV + log(Ft(t)) + (vt(t).^2).*(1./Ft(t));
    
    % Phi
    at_1p(t,1) = phip*atp(t-1);
    pt_1p(t,1) = (phip^2)*ptp(t-1) + Q;
    Ftp(t,1) = pt_1p(t,1) + H;
    vtp(t,1) = y(t) - at_1p(t);
    atp(t,1) = at_1p(t) + pt_1p(t).*(1./Ftp(t)).*vtp(t);
    ptp(t,1) = pt_1p(t) - (pt_1p(t).^2).*(1./Ftp(t));
    LogVp = LogVp + log(Ftp(t)) + (vtp(t).^2).*(1./Ftp(t));   
    
    % Q
    at_1q(t,1) = phi*atq(t-1);
    pt_1q(t,1) = (phi^2)*ptq(t-1) + Qq;
    Ftq(t,1) = pt_1q(t,1) + H;
    vtq(t,1) = y(t) - at_1q(t);
    atq(t,1) = at_1q(t) + pt_1q(t).*(1./Ftq(t)).*vtq(t);
    ptq(t,1) = pt_1q(t) - (pt_1q(t).^2).*(1./Ftq(t));
    LogVq = LogVq + log(Ftq(t)) + (vtq(t).^2).*(1./Ftq(t));    
end

Gr(1,1) = (LogVp - LogV)/ep;
Gr(2,1) = (LogVq - LogV)/ep;



