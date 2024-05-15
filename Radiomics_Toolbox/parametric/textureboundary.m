function [x,feats] = textureboundary(US, BW)
%SHAPE 此处显示有关此函数的摘要
% boundary feature
% most frequent value
% mean AT
% standard deviation AT
% skewness and kurtosis 偏度和峰度
% extract boundary regions

init_k = 4;
C = BW-imerode(BW,strel('disk',init_k));
A = sum(BW(:));
while sum(C(:)) / A < 0.2
    init_k = init_k + 2;
    C = BW-imerode(BW,strel('disk',init_k));
end
[x, featname] = statsfeats(US, C);
feats = cell(1);
feats_suff = {'B'};
for i = 1:length(feats_suff)
    for j = 1:10
        feats{(i-1)*10+j} = [featname{j} '_' feats_suff{i}];
    end
end
end