function [x,feats] = firstorder(PI, bw)
%UNTITLED5 计算边界、中心和整体的AT统计差异
%   此处显示详细说明
% PI 是时间分布图像, bw 是裁剪后的二值图像
[x_int, feat_int] = textureinterior(PI,bw); % 从bw中提取 5种， 每种里面有7个特征
US = double(PI);
Ig = US(logical(bw));  % 从PI图像中提取
% Ig(isnan(Ig))=[];
Ef = mode(Ig);
Em = mean(Ig);
Emd = median(Ig);
Etd = std(Ig);
% Eskw = skewness(Ig);
% Ek = kurtosis(Ig);
Eq = entropy(uint8(Ig));
Eq1 = quantile(Ig, 0.25); % 四分位数 Q1
Eq3 = quantile(Ig, 0.75); % 四分位数 Q3
Ei = Eq3 - Eq1;  % 箱线图间隔

x = [x_int  Ef Em Emd Etd Eq Eq1 Eq3 Ei];
feats = [feat_int,'freq','mean','median','std','entropy','quantile1','quantile3','quantileI'];
end

