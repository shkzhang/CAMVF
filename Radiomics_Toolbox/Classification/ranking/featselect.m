function [X, featn,idx] = featselect(X,Y,featn)
%% step 1 Feature selection based on MRMR criterion.
[ranked,score] = fscmrmr(X,Y);
score = score(ranked);
idx = ranked(1:find(score<1e-5,1));
featn = featn(idx);
X = X(:,idx);
if length(idx) > 10
    %% step 2
    c = cvpartition(Y,'k',10);
    opts = statset('Display','iter');
    [fs,~] = sequentialfs(@classf,X,Y,'cv',c,'options',opts);
    X = X(:,fs);
    featn = featn(fs);
    idx = idx(fs);
end
end

function [err] = classf(xtrain,ytrain,xtest,ytest)
[B,~,stats] = mnrfit(xtrain,ytrain);
[pihat,~,~] = mnrval(B,xtest,stats);
yfit = pihat(:,1);
yfit(yfit > 0.5) = 1;
yfit(yfit <= 0.5) = 2;
err = sum(ytest~=yfit)/ length(yfit);
end
