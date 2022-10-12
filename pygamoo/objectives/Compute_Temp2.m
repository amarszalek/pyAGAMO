%function Tmax = Compute_Temp2(ll, pp, bb, ss, I, AA, ls, lb)
function Tmax = Compute_Temp2(X)
%parpool('local',6);
addpath('C:\Users\Admin\Downloads\MULTIOBJECTIVE');
addpath('C:\Users\Admin\Downloads\MULTIOBJECTIVE\Factorize'); % load Factorize\*.m;
qv = 15000;
Tcz = 30;
%ll = 0.4;        %X(1);
%pp = 0.4;     %X(2);
%bb = 0.4;        %X(3);
%ss = 0.4;     %X(4);
%I = 1145;        %X(5);
%AA = 1800*1e-6;  %X(6);
alfa = 10;
%ls = 0.8;        %X(7);
%lb = 1.0;        %X(8);
sz = size(X);
Tmax = zeros(1,sz(1));
parfor i=1:sz(1)
    ll = X(i,1);
    pp = X(i,2);
    bb = X(i,3);
    ss = X(i,4);
    I = X(i,5);
    AA = X(i,6);
    ls = X(i,7);
    lb = X(i,8);
    [~, T] = FEM_Code(qv,Tcz,alfa,ll,ss,pp,bb,I,AA, ls, lb);
    %[~, Tmax(i)] = FEM_Code(qv,Tcz,alfa,ll,ss,pp,bb,I,AA, ls, lb);
    Tmax(i) = T;
end
%[~, Tmax] = FEM_Code(qv,Tcz,alfa,ll,ss,pp,bb,I,AA, ls, lb);