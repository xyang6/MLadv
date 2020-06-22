addpath('help_functions')

clear;clc
load('Data/LME.mat')
load('Pre_trained/LME_train.mat')

%% Setting
[data.N,pars.p] = size(X);
data.N_total  = 20;
data.prop_tr  = 0.2;
data.prop_val = 0.4;
pars.k        = 3;    %k used in kNN classification
pars.v        = 10;   %number of impostors
pars.max_iter = 5000; %maximum iter
pars.eta      = 1e-7; %criterion for early stopping: change in parameter
pars.alpha_3  = 0.1;  %initial learning rate
pars.alpha_1  = 0.99; %exponential decay parameter
pars.alpha_2  = 0.01; %exponential decay parameter

pars.L          = cell(data.N_total,1);
pars.M          = cell(data.N_total,1);
Log.obj_trace   = zeros(pars.max_iter,data.N_total);
Log.tri_trace   = zeros(pars.max_iter,data.N_total);
Log.ptb_trace   = zeros(pars.max_iter,data.N_total);
Log.norm_trace  = zeros(pars.max_iter,data.N_total);


%% training&test split
data = data_split(data);


%% Training and test
Acc = zeros(data.N_total,1);
for i_round = 1:data.N_total
    fprintf('round = %d \n',i_round);pars.trace = 0;
    X_train = X(data.Idx_train{i_round},:);
    Y_train = Y(data.Idx_train{i_round});
    X_test  = X(data.Idx_test{i_round},:);
    Y_test  = Y(data.Idx_test{i_round});
    
    %training stage
    Par_opt = pars.Par_opt(i_round,:);
    pars.mu = Par_opt(1);pars.lambda = Par_opt(2);pars.tau = Par_opt(3);pars.eps = Par_opt(4);
    pars.i_round = i_round; 
    [L,pars,Log] = ml_adv(X_train,Y_train,pars,Log);
    
    %test stage
    testerr      = knncl(L,X_train',Y_train',X_test',Y_test',pars.k,'train',0);
    Acc(i_round) = 1-testerr(end);
    
    clear X_train Y_train X_test Y_test L testerr Par_opt
end;clear i_round;
pars = rmfield(pars,'i_round');

clc;fprintf('Final result: mean(acc) = %.4f, std(acc) = %.4f\n',mean(Acc),std(Acc));

