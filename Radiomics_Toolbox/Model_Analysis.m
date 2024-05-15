clc;clear;
RUN_ME_FIRST;
warning('off')
%% read feat and label
[hccdata, ~, ~] = xlsread('../data/hcc.xlsx');
[iccdata, featn, ~] = xlsread('../data/icc.xlsx');
X = [hccdata(:,2:end);iccdata(:,2:end)];featn = featn(1,2:end);
%% remove null feats
nullid = sum(ismissing(X))~= 0;
X(:,nullid) = [];
featn(nullid) = [];
Y = [ones(size(hccdata,1),1); 2 * ones(size(iccdata,1),1)];
w2 = size(hccdata,1) / length(Y);
w1 = size(iccdata,1) / length(Y);
[X,~,~] = statnorm(X);
[ranked,score] = fscmrmr(X,Y);
score = score(ranked);
idx = ranked(1:find(score<1e-4,1));
featn = featn(idx);
Partition_Time = 5;
Partition_CV = 5;
cvo = cvpartition(Y,'KFold',Partition_CV);
for partition_time = 1:Partition_Time
    if partition_time ~= 1
        cvo = repartition(cvo);  % Repartition data for cross-validation
    end
    for partition_cv=1:Partition_CV
        train_index = cvo.training(partition_cv);
        test_index = cvo.test(partition_cv);
        Xtr = X(train_index, :); Xte = X(test_index, :);
        Ytr = Y(train_index); Yte = Y(test_index);
        response = zeros(length(Yte),1) * nan;
        %% feature selection
        [Xtr, selectfeat,idx] = featselect(Xtr,Ytr,featn); % Feature selection model
        Xte = Xte(:, idx);
        [B,dev,stats] = mnrfit(Xtr,Ytr);
        [pihat,dlow,hi] = mnrval(B,Xte,stats);
        response(pihat(:,1) > 0.5) = 1; 
        response(pihat(:,1) <= 0.5) = 2;
        cp = classperf(response, Yte);
    end
end












