clear
clc
addpath(genpath(pwd))

load  BUAA.mat % bbcsport msrc maxiter = 4
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});
k = 15;     % Msrc = 20 
pho = 1;    % 0.1;0.5;1
alpha_list = [0.001,0.1,0.5,1,3,5];
lambda_list = [0.001,0.1,0.5,1,3,5];
maxIter = 5; % bbcsport msrc maxiter = 4

for i = 1:length(alpha_list)
    for j = 1:length(lambda_list)
        alpha = alpha_list(i);
        lambda = lambda_list(j);
        [result(i,j)] = CTGL(data,truelabel,k,alpha,lambda,pho,maxIter);
        acc(i,j) = result(i,j).ACC
    end
end

%[FinalResult,V,L,G,A,W] = CTGL(data,truelabel,k,alpha_list(3),lambda_list(3),pho,maxIter);
