
function [Tim] = convertI(T, filenames, rect, basepath)
bw = cell(1, length(T));
% 遍历每一张图像
for i = 1:length(T)
    im = imread([basepath '\' filenames{i}]);
    im = imresize(im, [819, 1456]);
    im = imcrop(im, rect);  % size = 195 * 255 * 3
    % 将图像 im 转换为双精度类型，并将每个通道展平为一维数组
    im_flatten = reshape(double(im), [], 3); %  size = 49725 * 3
    % 计算在 RGB 通道上的标准差，并将结果大于 5 的像素标记为白色，形成二值图像 bw{i}
    bw{i} = reshape(std(im_flatten,0,2) > 5, size(im, 1), size(im, 2));  % size = 195 * 255
end
bwNew = cell(1, length(T));
bwNew{1} = bw{1};
for j = 1:length(T)-1
    % 计算当前图像与前一个图像之间的差异，生成新的二值图像
    bwNew{j+1} = logical(bw{j+1} - bw{j});
%     imshow(bwNew{j+1});figure;
end
Tim = zeros(size(im,1),size(im, 2));  % size = 195 * 255
for k = 1:length(T)
    % Tim 图像中的像素值表示对应位置的像素所对应的时间值
    Tim(bwNew{k}) = T(k);
end
% 部分无灌注区域， 原像素值为0的地方设置为 90
Tim(Tim==0) = 90;
% imshow(im);figure;
% imagesc(Tim);colormap turbo;
end

