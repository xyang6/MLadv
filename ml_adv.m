function [L,pars,Log] = ml_adv(X,Y,pars,Log)

%INPUT
%   X: (N*P)
%   Y: (N*1)
%   pars: hyperparameters
%   Log:  a log file which saves the output from the training stage
%OUTPUT
%   L: L'*L gives the Mahalanobis distance M; L' is the the projection matrix

%objective function:
%      mu/|R|*1{d2_M(x_i,x_l)<=d2_M(x_i,x_j)+tau}*sum[tau+d2_M(x_i,x_j)-d2_M(x_i,x_l)]_+
% +(1-mu)/|R|*1{d2_M(x_i,x_j)> d2_M(x_i,x_l)+tau}*sum[d2_M(x_i',x_j)-d2_M(x_i',x_l)]_+
% +    lambda*|M|_1
%dx: |dx|_infty = 1, |dx|_1 = eps


%% initialization
%construct triplets
Triplet = generate_knntriplets(X,Y,pars.k,pars.v)';
X_imp   = X(Triplet(3,:),:);
X_tar   = X(Triplet(2,:),:);
X_in    = X(Triplet(1,:),:);
T_ij    = X_in  - X_tar;
T_il    = X_in  - X_imp;
T_lj    = X_imp - X_tar;
n_T     = size(Triplet,2);
clear Triplet X_imp X_tar X_in

%initialize paramters
p     = pars.p;
M     = eye(p);
tau   = pars.tau;
iter  = 1;
n_max = pars.max_iter;
eta   = pars.eta;
eps   = pars.p-pars.eps;

%record
O_trace = zeros(n_max,1);
T_trace = zeros(n_max,1);
D_trace = zeros(n_max,1);
M_trace = zeros(n_max,1);
M_norm  = zeros(n_max,1);

%objective, gradient (T)
DM_T    = sum(T_ij.^2,2) - sum(T_il.^2,2);
T_val   = tau + DM_T;
slack_new = (T_val>=0); slack_old = slack_new;
T_12    = T_ij(slack_new,:); T_13 = T_il(slack_new,:);
Grad_T  = T_12' * T_12 - T_13'*T_13;

%objective, gradient (D)
Idx_dx  = (~slack_new);
Dx      = T_lj(Idx_dx,:);n_dx = sum(Idx_dx);
[~,Idx] = sort(abs(Dx),2);
Dx(sub2ind([n_dx,p],repmat((1:n_dx)',eps,1),vec(Idx(:,1:eps))))=0;
Dx      = sign(Dx);
D_ij    = T_ij(Idx_dx,:) + Dx;
D_il    = T_il(Idx_dx,:) + Dx;
DM_D    = sum(D_ij.^2,2) - sum(D_il.^2,2);
slack_D = (DM_D>=0);
D_ij    = D_ij(slack_D,:); D_il = D_il(slack_D,:);
Grad_D  = (D_ij' * D_ij - D_il' * D_il);
clear Idx_dx n_dx Idx D_ij D_il Dx

%calculate weights
par_T  = pars.mu/n_T;
par_D  = (1-pars.mu)*min(1,norm(Grad_T,'fro')/norm(Grad_D,'fro'))/n_T;
par_M  = pars.lambda;
%calculate objective function
T_trace(iter) = sum(T_val(slack_new)) *par_T;
D_trace(iter) = sum(DM_D(slack_D)) *par_D;
M_trace(iter) = pars.p *par_M;
O_trace(iter) = T_trace(iter) + D_trace(iter) + M_trace(iter);
clear DM_T DM_D slack_D


%% Metric learning
alpha_1 = pars.alpha_1; alpha_2 = pars.alpha_2;
alpha_0 = pars.alpha_3*sqrt(pars.p)/ (norm(Grad_T,'fro')*par_T + norm(Grad_D,'fro')*par_D) / exp(-(1+alpha_2)*alpha_1) ;

while iter < n_max
    alpha = alpha_0*exp(-(1+alpha_2*iter)*alpha_1);
    
    %Calculate gradient and Perform gradient descent
    Grad_T = calculate_grad(Grad_T,T_ij,T_il,slack_new,slack_old);
    Grad   = Grad_T*par_T + Grad_D*par_D;
    M      = M - alpha*Grad;
    
    %Proximal mapping
    thresh = par_M * alpha;
    M(M<=thresh & M>=-thresh) = 0;
    M(M> thresh) = M(M> thresh) - thresh;
    M(M<-thresh) = M(M<-thresh) + thresh;
    
    %Projection onto PSD cone
    [V,D]  = eig(M);
    D(D<1e-10) = 0;
    M      = V * D * V';
    M      = (M+M')/2;
    if all(M(:)==0);break;end
    
    %calculate objective function (T,norm)
    iter  = iter + 1;
    T_val = tau + sum((T_ij*M).*T_ij,2) - sum((T_il*M).*T_il,2);
    slack_old = slack_new; slack_new = (T_val>=0);
    T_val = sum(T_val(slack_new)) *par_T;
    T_trace(iter) = T_val;
    M_val = sum(abs(M(:))) *par_M;
    M_trace(iter) = M_val;
    
    %calculate objective function and gradient (dx)
    Idx_dx  = (~slack_new);
    Dx      = T_lj(Idx_dx,:)*M;n_dx = sum(Idx_dx);
    [~,Idx] = sort(abs(Dx),2);
    Dx(sub2ind([n_dx,p],repmat((1:n_dx)',eps,1),vec(Idx(:,1:eps))))=0;
    Dx      = sign(Dx);
    D_ij    = T_ij(Idx_dx,:) + Dx;
    D_il    = T_il(Idx_dx,:) + Dx;
    DM_D    = sum((D_ij*M).*D_ij,2) - sum((D_il*M).*D_il,2);
    slack_D = (DM_D>=0);
    D_ij    = D_ij(slack_D,:); D_il = D_il(slack_D,:);
    Grad_D  = (D_ij' * D_ij - D_il' * D_il);
    D_val   = sum(DM_D(slack_D)) *par_D;
    D_trace(iter) = D_val;
    clear Idx_dx n_dx Idx D_ij D_il Dx slack_D DM_D
    
    %calculate overall objective function
    o_val         = T_val + D_val + M_val;
    O_trace(iter) = o_val;
    
    %check convergence
    if iter > 50 && max(abs(diff(O_trace(iter-4:iter)))) < eta*o_val
        break
    end
end

L = sqrt(D) * V';

pars.L{pars.i_round}           = L;
pars.M{pars.i_round}           = M;
Log.obj_trace(:,pars.i_round)  = O_trace;
Log.tri_trace(:,pars.i_round)  = T_trace;
Log.ptb_trace(:,pars.i_round)  = D_trace;
Log.norm_trace(:,pars.i_round) = M_trace;
Log.iter(pars.i_round)         = iter;
pars = rmfield(pars,{'mu','lambda','tau'});
if iter == n_max
    fprintf('metric learning stops without convergence \n')
end

end
