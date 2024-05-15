function [rect] = cropI(bw,r)
se = strel('disk',r);  % 创建一个圆形结构元素 se，其半径为 r，这个结构元素将用于膨胀操作。
enlarged_bw = imdilate(bw,se);  % 对二值图像 bw 进行膨胀操作
% 计算enlarged_bw 中的区域属性，指定的属性是面积（Area）
stats = regionprops(enlarged_bw, 'Area', 'BoundingBox');
% 获取所有区域的面积
areas = [stats.Area];
% 找到面积最大的区域的索引
[maxArea, maxIndex] = max(areas);
% 获取面积最大的区域的边界框
rect = round(stats(maxIndex).BoundingBox);
end

