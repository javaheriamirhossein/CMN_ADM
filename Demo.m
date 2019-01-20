% -------------------------------------------------------------------------
%
% This is a simple demo file to test the CMN-ADM algorithm
% This script file is a modified version of the one in the following 
% sourcecode: https://github.com/FWen/Lp-Robust-CS.git
% 
% -------------------------------------------------------------------------

clear all; 
close all;
clc;

N = 128;
K = ceil(8*N/128);
M = ceil(50*N/128);

A = randn(M, N);
A = orth(A')';

amp_ratio = 10;
x  = amp_ratio*SparseVector(N,K); % the generated sparse signal
y0 = A*x;


alpha = [0.75];
gamma = [1e-2];

% ------------- Generating noisy signal -----------------
% noise = randn(M,1);
noise = stblrnd(alpha,0,gamma,0,M,1);
y = y0 + noise;

% -------------------------------------------------------

% the value of lambda is the same for all algorithms
lambda = 0.05*norm(A'*y,'inf');

% --------------------------------------------------------


figure(1);
subplot(3,4,1);plot(1:length(x),x);xlim([1 length(x)]);
title(['Recovery of sparse signal corrupted with S\alphaS noise ', '(\alpha=', num2str(alpha), ' \gamma=',num2str(gamma),')'],'FontSize', 13);
xlabel('(a) Test signal (x)');set(gcf,'outerposition',get(0,'screensize'));
subplot(3,4,2);plot(1:length(y0),y0);
xlabel('(b) Clean measurements signal (Ax)');

subplot(3,4,3);plot(1:length(y),noise);
xlabel('(d) noise (n)');

subplot(3,4,4);plot(1:length(y),y);
xlabel('(c) Corrupted signal (Ax+n)');


%% -------- Comparing different algorithms --------------------------------

%----Lq-min----------------------------------------
t0 = tic;
[x_Lq] = lq(y, A, 1./(2:11),[0; 0.1; 0.2],2);
relerr_LqMin = norm(x_Lq - x)/norm(x);
disp(sprintf('Lq-min:\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,5);plot(1:length(x_Lq),x_Lq);
xlabel(['(e) Lq-Min, RelErr=', num2str(relerr_LqMin,'%10.3f')]); xlim([1 length(x)]);
figure(2); subplot(2,4,1);plot(1:length(x_Lq),x_Lq-x);
xlabel(['(b) Lq-Min, RelErr=', num2str(relerr_LqMin,'%10.3f')]);
title('Recovery error');
xlim([1 length(x)]);



%----Yall1-----------------------------------------
t0=tic;
[x_Yall1] = YALL1_admm(A, y, lambda, 1, x);
relerr_Yall1 = norm(x_Yall1 - x)/norm(x);
disp(sprintf('Yall1:\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,6);plot(1:length(x_Yall1),x_Yall1);
xlabel(['(f) Yall1, RelErr=', num2str(relerr_Yall1,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,2);plot(1:length(x_Yall1),x_Yall1-x);
xlabel(['(c) Yall1, RelErr=', num2str(relerr_Yall1,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');



%----CMN-ADM(0,1,1)--------------------------------
t0=tic;
[x_CMN_ADM] = CMN_ADM(A, y, 0, 1, 1);
relerr_CMN_ADM = norm(x_CMN_ADM - x)/norm(x);
disp(sprintf('CMN-ADM(0,1,1):\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,7);plot(1:length(x_CMN_ADM),x_CMN_ADM);
xlabel(['(f) CMN-ADM(0,1,1), RelErr=', num2str(relerr_CMN_ADM,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,3);plot(1:length(x_CMN_ADM),x_CMN_ADM-x);
xlabel(['(c) CMN-ADM(0,1,1), RelErr=', num2str(relerr_CMN_ADM,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');



%----CMN-ADM(0,1,2)--------------------------------
t0=tic;
[x_CMN_ADM] = CMN_ADM(A, y, 0, 1, 2);
relerr_CMN_ADM = norm(x_CMN_ADM - x)/norm(x);
disp(sprintf('CMN-ADM(0,1,2):\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,8);plot(1:length(x_CMN_ADM),x_CMN_ADM);
xlabel(['(f) CMN-ADM(0,1,2), RelErr=', num2str(relerr_CMN_ADM,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,4);plot(1:length(x_CMN_ADM),x_CMN_ADM-x);
xlabel(['(c) CMN-ADM(0,1,2), RelErr=', num2str(relerr_CMN_ADM,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');



%---Huber-FISTA------------------------------------
t0 = tic;
epsilon = 0.1; nu_est = 0.05;
[x_HFISTA]  = fista_robust_cs(A, y, lambda, epsilon, nu_est, 0, x_Yall1);
relerr = norm(x_HFISTA - x)/norm(x);
disp(sprintf('Huber-FISTA: elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,9);plot(1:length(x_HFISTA),x_HFISTA);
xlabel(['(g) Huber-FISTA, RelErr=', num2str(relerr,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,5);plot(1:length(x_HFISTA),x_HFISTA-x);
xlabel(['(d) Huber-FISTA, RelErr=', num2str(relerr,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');


%----BP-SEP----------------------------------------
t0=tic;
A1 = [A eye(M)];
A1 = A1/sqrt(2);
y1  = y/sqrt(2);
[x_bp] = admm_BPSEP(A1, y1, lambda, x, 10);
x_bp = x_bp(1:N);
relerr_BPSEP = norm(x_bp(1:N) - x)/norm(x);
disp(sprintf('BP-SEP: \t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,10);plot(1:length(x_bp),x_bp);
xlabel(['(j) BP-SEP, RelErr=', num2str(relerr_BPSEP,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,6);plot(1:length(x_bp),x_bp-x);
xlabel(['(f) BP-SEP, RelErr=', num2str(relerr_BPSEP,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');



%--Lp-ADM (p=1)------------------------------------
t0 = tic;
[x_lpl1]   = l1_lp_admm(A, y, lambda, 1, x, 1e2);
relerr = norm(x_lpl1 - x)/norm(x);
disp(sprintf('Lp-ADM (p=1):  elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,11);plot(1:length(x_lpl1),x_lpl1);
xlabel(['(d) Lp-ADM (p=1), RelErr=', num2str(relerr,'%10.3f')]);
xlim([1 length(x)]);
figure(2); subplot(2,4,7);plot(1:length(x_lpl1),x_lpl1-x);
xlabel(['(a) Lp-ADM (p=1), RelErr=', num2str(relerr,'%10.3f')]);
xlim([1 length(x)]);title('Recovery error');
set(gcf,'outerposition',get(0,'screensize'));



%----Lp-ADM (p=0.5)--------------------------------
t0 = tic;
[x_L1Lp05] = l1_lp_admm_ac(A, y, lambda, 0.5, x, 1e4, x_lpl1);
relerr = norm(x_L1Lp05 - x)/norm(x);
disp(sprintf('Lp-ADM (p=0.5): elapsed time is %.3f seconds',toc(t0)));
figure(1); subplot(3,4,12);plot(1:length(x_L1Lp05),x_L1Lp05);
xlabel(['(g) Lp-ADM (p=0.5), RelErr=', num2str(relerr,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,8);plot(1:length(x_L1Lp05),x_L1Lp05-x);
xlabel(['(d) Lp-ADM (p=0.5), RelErr=', num2str(relerr,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');





savefig(figure(1),['Fig1_',num2str(alpha),'_',num2str(gamma),'.fig']);
savefig(figure(2),['Fig2_',num2str(alpha),'_',num2str(gamma),'.fig']);
save('denoising_data.mat');
