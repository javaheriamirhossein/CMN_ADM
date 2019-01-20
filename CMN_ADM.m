function [ x,rel_err,time_iter ] = CMN_ADM( A,y,p_s,p_f,q,x0,Opt )

%-----------------------------------------------------------------------------------------------------------
% CMN_ADM algorithm to solve 
%
%       minimize CMN[(A*x-y)/sigma_n] + mu*||x||_1,
%
%   where A is the measurement matrix, y is the observed signal purturbed with impulsive noise and x is
%   the sparse coefficient vector
%-------------------------------------------------------------------------
%
% INPUT
%   A           : mxn  measurement matrix
%   y           : mx1 vector of observed values
%   p_s         : the lower limit of the CMN (integral over p-norms)
%   p_f         : the upper limit of the CMN (integral over p-norms)
%   q           : the q-norm used as surrogate: either 1 or 2
%   x0          : true (initial) value of x
%   Opt         : struct variable of options 
%   Opt.maxIter : max iteration limit of the algorithm
%   Opt.report  : if set to 1, the algorithm computes the time and relative error of reconstruction in each
%                 iteration
%   Opt.mu      : the value of mu (regularizing parameter)
%
%
% OUTPUT
%   x           : nx1 vector of sparse coefficients
%   rel_err     : vector values of relative error in each iteration
%   time_iter   : vector values of time taken in each iteration
%
% USAGE EXAMPLES
%   [x] = CMN_ADM(A,y);
%   [x,rel_err,time_iter] = CMN_ADM(A,y,0,1,2,Opt);
%
% ------------------------------------------------------------------------
% AUTHOR:    Amirhossein Javaheri <javaheriamirhosein@gmail.com>
% UPDATE:    Apr 8 2018
%
% Reference:
% A. Javaheri, H. Zayyani, F. Marvasti and M.A.T. Figueiredo, " Robust Sparse Recovery
% in Impulsive Noise via Continuous Mixed Norm", IEEE Signal Processing Letters, vol. 25, 2018.
%-------------------------------------------------------------------------------------------------------------
[m,n] = size(A);


if nargin<7
    Opt.report = 0; end
if nargin<6
    x0 = zeros(n,1); end
if nargin<5
    q = 2; end
if nargin<4
    p_f = 1; end
if nargin<3
    p_s = 0; end
  
if p_s>p_f
    error('p_s should be less than p_f'); end
if p_f>q
    error('p_f should be less than q'); end

if ~isfield(Opt,'maxIter')
    Opt.maxIter=100; end
if ~isfield(Opt,'report')
    Opt.report =0; end


% -------------------------------------------------------------------------------------------
At = A';
AtA = At*A;
maxIter = Opt.maxIter;


Niteration = maxIter;
rel_err = NaN(1,Niteration);     
time_iter = NaN(1,Niteration);   

                
norm_x0 = norm(x0);          
%---------------- Parameters -----------------------------------------------------------------

xsi = 0.95;
ratio = 0.1;
Lambda0 = 2;
sigma = 1;
sigman = 1;
mu_min = 0.5;         % In case there is no noise choose tau_min = 1e-5
eps = 0.01;
beta = 1.1;           % this parameter is used in backtracking solution of x-step subproblem
sigma_inv = 1/sigma;
sigmn_inv = 1/sigman;
stop_tol = 1e-5;

%------------ Initialization ------------------------------------------------------------------

At_y = At*y;
x = zeros(n,1);
z = -y;
eta = zeros(m,1);

if ~isfield(Opt,'mu')
    Opt.mu =ratio *norm(At_y,'inf'); end

mu = Opt.mu;

% ----------------------------------------------------------------------------------------------
t0 = tic;

for iter =1: maxIter
    
    % ---------- backtracking to solve for x (x update step) --------------------------
    c = At_y*sigmn_inv+ At*(z-eta*sigma_inv);
    
    temp = AtA*x*sigmn_inv - c ;
    stop_backtrack = 0 ;
    
    while ~stop_backtrack
        
        gk = x - (1/Lambda0)*sigmn_inv*temp ;
        thr = mu/Lambda0*sigma_inv;
        xp = Threshold(gk,thr) ;
        
        temp1 = norm(A*(x-xp))^2 ;
        temp2 = Lambda0*norm(x-xp)^2;
        
        if temp1 <= temp2
            stop_backtrack = 1 ;
        else
            Lambda0 = Lambda0*beta ;   % Lambda0 is increased untill it satisfies surrogate property 
        end
    end
    
    x = xp;
    mu = max(xsi*mu,mu_min) ;          % continuation on the regularizing parameter
    
    
    % ---------- (Computing error)----------------------------------------------------
    
    if Opt.report
        time_iter(iter) = toc(t0);
        rel_err(iter) = norm(x0 - x)/norm_x0;
    end
       
    % ---------- (z update step) ------------------------------------------------------
    
    z_ab = abs(z)+eps;
    log_z_ab = log(z_ab);
    z_ab_q = z_ab.^q;
    
    num = z_ab.^p_f.*(p_f*log_z_ab-1)- z_ab.^p_s.*(p_s*log_z_ab-1);
    denum = (p_f-p_s)*q*z_ab_q.*log_z_ab.^2;
    phi_z = num./denum;
    
    
    temp = (A*x-y)*sigmn_inv;
    b = temp +eta*sigma_inv;
    
    switch q
        case 1
            thr = phi_z*sigma_inv;
            z = Threshold(b,thr);
        case 2
            w = 1+2*sigma_inv*phi_z;
            z = b./w;
        otherwise
            error('q should be either 1 or 2');
    end
    % ----------------------------------------------------------------------------------

    if norm(temp-z)<stop_tol
        break;
    end

    eta = eta+sigma*(temp-z);
    
end

if Opt.report
    toc(t0);
end

end

function [shat,Supp] = Threshold(s,thr)
    Supp=(abs(s)>=thr);
    shat = sign(s).*(abs(s)-thr).*Supp;
end
