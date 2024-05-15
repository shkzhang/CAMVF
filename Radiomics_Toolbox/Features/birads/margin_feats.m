% MARGIN_FEATS Compute BI-RADS margin features.
%   [X,FEAT] = MARGIN_FEATS(BW) computes margin BI-RADS features according 
%   to the BI-RADS lexicon defined for breast ultrasound, where BW is the 
%   binary shape of the lesion:
%   
%   BI-RADS feature         Quantitative feature
%   ---------------         ----------------------------  
%   Margin
%                           Number of undulations (U)%起伏数
%                           Angularity feature (A)%角度特征是乳腺超声图像中的一种边缘特征，用于描述病变边缘的角度特点
%                           Sum of U and A
%   
%   Example:
%   -------
%   load('BUS01.mat');   
%   [x,feat] = margin_feats(Smanual);
%
%   See also BIRADS_FEATS BOUND_FEATS ECHO_FEATS ORIENT_FEATS SHAPE_FEATS
%
%
%   References:
%   ----------
%   W. K. Moon, C. M. Lo, et al. "Quantitative ultrasound analysis for 
%   classification of BI-RADS category 3 breast masses," J Digit Imaging,
%   vol. 26, pp. 1091-1098, 2013.
%
%   W.-C. Shen, R.-F. Chang, W. K. Moon, Y.-H. Chou, C.-S. Huang, "Breast 
%   ultrasound computer-aided diagnosis using bi-rads features," Acad Radiol,
%   vol. 14, no. 8, pp. 928-939, 2007.

% ------------------------------------------------------------------------
%   Cinvestav-IPN (Mexico) - LUS/PEB/COPPE/UFRJ (Brazil)
%   MARGIN_FEATS Version 1.0 (Matlab R2014a Unix)
%   December 2016
%   Copyright (c) 2016, Wilfrido Gomez Flores
% ------------------------------------------------------------------------

function [x,feats] = margin_feats(BW)
% Crop
[y,x] = find(BW);%非零坐标
xmn = min(x);
xmx = max(x);
ymn = min(y);
ymx = max(y);
BW2 = BW(ymn:ymx,xmn:xmx);%裁剪
BW2 = padarray(BW2,[1 1],0,'both');%边缘0填补一周
% Compute distance map，计算距离映射
D = bwdist(~BW2);%计算非零像素到离它最近的0像素的距离
% Inscribed circle distance map，计算内切圆的距离映射
[M,N] = size(BW2);%尺寸
[y,x] = find(D==max(D(:)));%最大像素距离的点
xc = mean(x); yc = mean(y);%计算均值
[X,Y] = meshgrid(1:N,1:M);%BW2尺寸大小
C = sqrt((X-xc).^2 + (Y-yc).^2);%计算内切圆的距离映射
% Get lobules with the maximum inscribed circle，获取具有最大内切圆的小叶
r = max(D(:))+1; % Radius
BW3 = xor(~BW2,C>=r);%原始图像BW2中0像素到内切圆心的距离大于等于半径r,1像素小于r
BW3 = bwareaopen(BW3,2);%删除小于等于2个像素的连通组件
D2 = bwdist(~BW3);
BW3(D2<2) = 0;%小于2的距离的像素设置为零
% Count undulations without considering slighter undulations
L = bwlabel(BW3);%标识不同的连通组件
nl = max(L(:));%连通数
U = 0;
for i = 1:nl
    idx = L==i;%i所在为1，其余为0
    D3 = bwdist(~idx); % Interior distances，非i区域（非零元素）到i区域的0元素的距离
    if max(D3(:)) > 3 %如果内部距离图像的最大距离大于3
        U = U+1; % Its a lobule，% 增加波状特征的数量
    end
end
% Compute angularity feature
% Skeleton
S = bwmorph(BW2,'skel','inf'); % skeleton(BW2, D, 10, 1);骨架提取，inf指定无限次迭代，确保完整骨架
BW4 = bwareaopen(and(S,~(C<r)),5);% 骨架中，排除半径小于 r 的像素，再排除小于5
L2 = bwlabel(BW4);%连通组件进行标记
A = max(L2(:)); % Angularity，连通数
MUA = U+A;
% Features
x = [U A MUA];
feats = {'mMU','mMA','mMUA'};