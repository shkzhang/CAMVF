function [x,feats] = NonPerfusion(US, BW)
us = US;
stats = regionprops(BW,'Area');
bw = us == 90; % 找到无灌注的区域
A = sum(sum(bw)); % 按照行进行sum，然后在按照列进行sum
if A > stats.Area * 0.05
    x = 1;
    x = [x A/stats.Area];
else
    x = [0 0];
end
feats = {'NonPerfusion', 'NonPerfusionArea'};
end