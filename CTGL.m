function [FinalResult,V,Lf,G,A,W] = CTGL(data,truelabel,k,alpha,lambda,pho,maxIter)
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});
mu = 2;
gamma = 1;

%% Dataset Normalization and Initialization 
for i = 1:numView
    data{i} = data{i}./repmat(sqrt(sum(data{i}.^2,1)),size(data{i},1),1);  %normalized
    G{i} = constructW_PKN(data{i},k);   
    NanIdx = isnan(G{i});
    G{i}(NanIdx) = 0;
    %G{i} = G{i} - diag(diag(G{i}));
end

[~,~,label,~,W] = Fusion(G, nCluster, numView);
A = G;  % Initialize A by given graph G 

for v = 1:numView+1
    P{v} = zeros(n,n);
    Q{v} = zeros(n,n);  %multiplier
end

Winit = W;

sX = [n, n, numView+1]; 
%maxIter = 4;

for iter = 1:maxIter

    %% Unified Graph learning W
    U = zeros(n,n);
    for v = 1:numView
        U = U + A{v};
    end
    B = P{numView+1} - Q{numView+1}/pho;
    ed = L2_distance_1(U, U);
    for j = 1:n
        ad = (pho*B(j,:)-ed(j,:))/(2*alpha+pho);
        Wtemp(j,:) = EProjSimplex_new(ad);
    end
    W = Wtemp;
    W = (W + W')/2;
    W = abs(W);
    %W = W - diag(diag(W));

    A{numView+1} = W;
    %% Low-Rank Tensor Completion
    A_tensor = cat(3, A{:,:});
    Q_tensor = cat(3, Q{:,:});
    a = A_tensor(:);
    q = Q_tensor(:);
    [j, ~] = wshrinkObj(a+1/pho*q,lambda/pho,sX,0,3);
    P_tensor = reshape(j, sX); 
    for i = 1:numView+1
        P{i} = P_tensor(:,:,i);
    end

    % Update multiplier Q
    for i = 1:numView+1
        Q{i} = Q{i}+pho*(A{i}-P{i});
    end
    
    % Update parameter
    pho = pho*mu;

    D0 = diag(sum(W));
    L = D0 - W;
    I = eye(n);
    %% Subgraph Propagation A_{1,2,...,m}
    L = I*(1+pho) + L;
    Obj = [];
    ObjTemp = 0;
    for v = 1:numView
        B = P{v} - Q{v}/pho;
        B = (B + G{v})/gamma;
        for i = 1:n
            index = find(W(i,:)>0);
            bi = B(i,index);
            b = 2*bi;
            [ai,objind] = fun_alm(L(index,index),b);
            A{v}(i,index) = ai';
        end
    end 
    
%     for i = 1:numView+1
%         A{i} = A{i} - diag(diag(A{i}));
%     end

    [Lf,V,predictLabel] = FusionSum(A, nCluster, numView+1);
    %[Lf,V,predictLabel] = FusionSum(A, nCluster, numView);
    FinalResult = ClusteringMeasure_new(truelabel{1}, predictLabel);
end

