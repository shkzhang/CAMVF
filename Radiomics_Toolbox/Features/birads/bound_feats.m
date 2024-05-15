% BOUND_FEATS Compute BI-RADS boundary features.
%   [X,FEAT] = BOUND_FEATS(I,BW) computes BI-RADS boundary according to the
%   BI-RADS lexicon defined for breast ultrasound, where I is the gray-scale 
%   image containing a lesion and BW is the binary shape of the lesion:
%   
%   BI-RADS feature         Quantitative feature
%   ---------------         ----------------------------
%   Boundary
%                           Abrupt interface at 10 pixels,图像边界的10像素内是否存在突变的接口
%                           Abrupt interface at 25% of mass area,质点区域的边界的25%处是否存在突变的接口
%                           Normalized radial gradient,标准化的径向梯度
%   
%   Example:
%   -------
%   load('BUS01.mat');   
%   [x,feat] = bound_feats(I,Smanual);
%
%   See also BIRADS_FEATS ECHO_FEATS MARGIN_FEATS ORIENT_FEATS SHAPE_FEATS
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
%   BOUND_FEATS Version 1.0 (Matlab R2014a Unix)
%   December 2016
%   Copyright (c) 2016, Wilfrido Gomez Flores
% ------------------------------------------------------------------------

function [x,feats] = bound_feats(I,BW)
I = double(I);
[M,N] = size(I);
% Compute distance map
Din  = bwdist(~BW);% U形
Dout = bwdist(BW); % Ω形
th = 10;
LB = mean(I((Dout<=th)&(Dout>0))) - mean(I((Din<=th)&(Din>0)));
%using the distance that represents 25% percent of the mass of the lesion
LB_25 = bound_percentage(I,BW,Din,Dout,0.25);
%using the distance that represents 50% percent of the mass of the lesion
LB_50 = bound_percentage(I,BW,Din,Dout,0.50);
%using the distance that represents 50% percent of the mass of the lesion
LB_100 = bound_percentage(I,BW,Din,Dout,1.00);
% Normalized radial gradient
sx = [-1 -2  0 2  1
      -4 -8  0 8  4
      -6 -12 0 12 6
      -4 -8  0 8  4
      -1 -2  0 2  1];
sy = sx';
gx = imfilter(I,sx,'replicate');%对输入图像I进行滤波，计算x方向的梯度
gy = imfilter(I,sy,'replicate');%对输入图像I进行滤波，计算y方向的梯度
[y,x] = find(BW);
xc = mean(x);
yc = mean(y);
[x,y] = meshgrid(1:N,1:M);
x = x - xc;
y = y - yc;
n = sqrt(x.^2+y.^2);
xu = x./n;
yu = y./n;%单位向量
F1 = gx.*xu + gy.*yu;%归一化径向梯度
F2 = sqrt(gx.^2+gy.^2);%模长
C = bwperim(BW);% 找到BW的边界像素
nrg = sum(F1(C))/sum(F2(C));
% Features
x = [LB LB_25 LB_50 LB_100 nrg];
feats = {'bLB10px','bLB25%','bLB50%','bLB100%','bNRG'};

%************************************************************************
function LB_1 = bound_percentage(I,BW,Din,Dout,fsz1)
k=1;% 找到一个合适的距离阈值 k，以满足所需的百分比
while (sum(BW(:))*fsz1)>sum(sum(Din<=k&Din>0)) &&  (sum(BW(:))*fsz1)>sum(sum(Dout<=k&Dout>0))
    k = k+1;
end
Win1  = Din<=k&Din>0;
Wout1 = Dout<=k&Dout>0;
LB_1 = mean(I(Wout1))-mean(I(Win1));%描述边界的性质