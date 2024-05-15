function [x,feats] = centroidnull(US,BW)
%CENTROIDNULL 此处显示有关此函数的摘要
%   此处显示详细说明
US = double(US);
sz=size(BW);
Pbw = regionprops(BW,'Area','Centroid');
xc = Pbw.Centroid(1); yc = Pbw.Centroid(2);
r0= sqrt(Pbw.Area * 0.15 / pi);      % 取中心区域 百分之15 的圆域
[xgrid, ygrid] = meshgrid(1:sz(2), 1:sz(1));
x = xgrid - xc;    
y = ygrid - yc;
C0 = x.^2 + y.^2 <= r0.^2;
Ig = US(logical(C0));
x = sum(ismissing(Ig));
stats = regionprops(BW, 'Area');
x = x / stats.Area;
feats = cellstr('central_fill');
end

