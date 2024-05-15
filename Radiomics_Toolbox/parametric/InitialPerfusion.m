function [x,feats] = InitialPerfusion(US, BW, T)
t0 = min(T) - 2;

bw_2 = US == T(1);
bw_4 = US == T(2);

S2 = regionprops(bw_2,'Area');
S2 = sum([S2.Area]);
S4 = regionprops(bw_4,'Area');
S4 = sum([S4.Area]);

S = size(US, 1) * size(US, 2);


ratio_2 = S2 / S;
ratio_4 = (S2 + S4) / S;
if ratio_4 > 0.55
    fast = 1;
else
    fast = 0;
end
feats = {'initial_T', 'initialArea_2', 'initialArea_4', 'fastEnhance'};
x = [t0, ratio_2, ratio_4, fast];
end