function [x,feats] = spatialstats(US,BW)
%   各个颜色区域中心坐标相对位置，二维坐标的均值和方差
Time = [0 2 4 6 10];
stats = regionprops(BW, 'BoundingBox', 'Area');
boxes = round(stats.BoundingBox);
area = stats.Area;
C0 = US == Time(1);
C2 = US == Time(2);
C4 = US == Time(3);
[Cr0, x0, ~] = findCord(C0,boxes,area);
[Cr2, x2, ~] = findCord(C2,boxes,area);
[Cr4, x4, featname] = findCord(C4,boxes,area);
x = [x0 Cr0 x2 Cr2 x4 Cr4];
feats = cell(1);
featname{5} = 'centroid_x';featname{6} = 'centroid_y';
feats_suff = {'C0','C2','C4'};
for i = 1:length(feats_suff)
    for j = 1:6
        feats{(i-1)*6+j} = [featname{j} '_' feats_suff{i}];
    end
end
end

function [Cr, x,feats] = findCord(C0,boxes,area)
C0 = bwmorph(C0,'clean');
C0 = bwmorph(C0,'close');
w = boxes(3);
h = boxes(4);
largest_bw = zeros(size(C0));
if sum(C0(:)) ~= 0
    L = bwlabel(C0);
    stats = regionprops(logical(C0));
    Ar = cat(1, stats.Area);
    Cr = cat(1, stats.Centroid);
    Cr = Cr(Ar == max(Ar),:); % 找到最大连通区域的标号
    ratios = max(Ar) / area;
    Cr(1) = Cr(1) / h;
    Cr(2) = Cr(2) / w;
    largest_bw(L == find(Ar == max(Ar))) = 1;
    [x,feats] = shape(largest_bw);
    x = [x ratios];
    feats = [feats 'ratio'];
else
    Cr = nan;
end
end

