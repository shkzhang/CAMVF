function [x, feats] = statsfeats(US,C0)
try
    Ig = US(logical(C0));
    % Ig(isnan(Ig))=[];
    Ef = mode(Ig); % 计算向量 Ig 中的众数
    Em = mean(Ig); % 众数
    % Emd = median(Ig);
    Etd = std(Ig); % 标准差
    % Eskw = skewness(Ig);
    % Ek = kurtosis(Ig);
    Eq = entropy(uint8(Ig)); % 熵值
    Eq1 = quantile(Ig, 0.25); % 四分位数 Q1
    Eq3 = quantile(Ig, 0.75); % 四分位数 Q3
    Ei = Eq3 - Eq1;  % 箱线图间隔
    x = [Ef Em Etd Eq Eq1 Eq3 Ei];
    feats = {'freq','mean','std', 'Entropy','quantile1','quantile3','quantileI'};
    % 特征名称 feats ，特征向量 x
end

