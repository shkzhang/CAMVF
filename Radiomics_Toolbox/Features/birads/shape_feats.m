% SHAPE_FEATS Compute BI-RADS features.
%   [X,FEAT] = SHAPE_FEATS(BW) computes BI-RADS shape features according to the
%   BI-RADS lexicon defined for breast ultrasound, where BW is the binary shape 
%   of the lesion:
%   
%   BI-RADS feature         Quantitative feature
%   ---------------         ----------------------------
%   Shape                   
%                           Normalized residual value
%                           Normalized radial lenght
%                           Overlap with equivalent ellipse (EE)
%                           Elliptic-normalized circumference
%                           Elliptic-normalized skeleton
%                           Long axis to short axis ratio of EE
%                           Compactness or roundness
%                           Shape class
%                           Proportional distance between edges
%                           Major and minor axis length of EE
%
%   Example:
%   -------
%   load('BUS01.mat');   
%   [x,feat] = shape_feats(Smanual);
%
%   See also BIRADS_FEATS BOUND_FEATS ECHO_FEATS MARGIN_FEATS ORIENT_FEATS
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
%   SHAPE_FEATS Version 1.0 (Matlab R2014a Unix)
%   December 2016
%   Copyright (c) 2016, Wilfrido Gomez Flores
% ------------------------------------------------------------------------

function [x,feats] = shape_feats(BW)
% Crop
[y,x] = find(BW);% 查找二值图像中非零像素的坐标
xmn = min(x);% 找到 x 坐标的最小值
xmx = max(x);
ymn = min(y);
ymx = max(y);
BW2 = BW(ymn:ymx,xmn:xmx);% 裁剪二值图像，将其限制在包含病变的最小矩形区域内
BW2 = padarray(BW2,[1 1],0,'both');% 在裁剪后的图像的边缘用零填充一圈
%---------------------------------------------------------------------
% Normalized Residual Value
BW_props = regionprops(BW2,'ConvexHull','Perimeter','Centroid','Area');%计算二值图像的一些属性
CH = roipoly(BW2,BW_props.ConvexHull(:,1),BW_props.ConvexHull(:,2));%利用连通组件的凸包坐标，获得裁剪图像的ROI
NRV = bwarea(xor(BW2,CH))/bwarea(CH);% 计算归一化残差值，非凸包/凸包
%---------------------------------------------------------------------
% NRL
[xnrl,fnrl] = nrl(BW);% 调用 nrl 函数计算一些与病变形状相关的特征
%---------------------------------------------------------------------
% Elipse equivalente
Pbw = regionprops(BW,'Area','Centroid','Perimeter');%面积，质心，周长
A = Pbw.Area;%病变面积
xc = Pbw.Centroid(1); yc = Pbw.Centroid(2);%病变的质心坐标
[y,x] = find(BW);% 查找二值图像中非零像素的坐标
[xx,yy] = meshgrid(1:size(BW,2),1:size(BW,1)); % 创建网格坐标
% Calcula los momentos de segundo orden del objeto binario original，计算二进制图像的原始二阶矩
Sxx = (1/A)*sum((x-xc).^2);
Syy = (1/A)*sum((y-yc).^2);
Sxy = (1/A)*sum((x-xc).*(y-yc));% x 和 y 之间的交叉二阶矩
% Calcula los coeficientes de la ecuacion general de la elipse
coef = (1/(4*Sxx*Syy - Sxy^2))*[Syy -Sxy;-Sxy Sxx];%系数矩阵 coef 是一个 2x2 的矩阵，其中包含了椭圆方程的系数
a = coef(1,1); b = coef(1,2); c = coef(2,2);%coef(2,1) 对应于椭圆方程中的系数 B
% Calcula la elipse eqivalente del objeto binario，二值化对象的等效椭圆
E = (a*(xx-xc).^2 + 2*b*(xx-xc).*(yy-yc) + c*(yy-yc).^2) < 1;
% Calcula el contorno y el perimetro de la elipse equivalente
E_props = regionprops(E,'Perimeter','MinorAxisLength','MajorAxisLength');% 周长，次轴、主轴长度
%---------------------------------------------------------------------
OR  = bwarea(BW&E)/bwarea(BW|E);	% Overlap，重叠度
%---------------------------------------------------------------------
ENC = E_props.Perimeter/Pbw.Perimeter; % ENC，计算等效椭圆的周长与病变周长的比值
S = bwmorph(BW,'skel','inf');% 对二值图像进行骨架化处理
S = bwareaopen(S,5);% 移除小的骨架组件
ENS = sum(S(:))/E_props.Perimeter; % ENS，计算等效椭圆的骨架长度与病变周长的比值
%---------------------------------------------------------------------
LS = E_props.MajorAxisLength/E_props.MinorAxisLength; %LS，长轴短轴比
Emax = E_props.MajorAxisLength;%长轴
Emin = E_props.MinorAxisLength;%短轴
%---------------------------------------------------------------------
R = 1 - ((4*pi*A)/(Pbw.Perimeter^2)); % Compactness，计算紧凑度
%---------------------------------------------------------------------
% Shape class
% Parametrizacion de los contornos de BW y elipse
junk = bwboundaries(BW);%找到边界坐标
cBW  = junk{1};%第一个对象的边界坐标
yBW  = cBW(:,1); xBW = cBW(:,2);%这两行代码将 cBW 中的 y 和 x 坐标分别存储在 yBW 和 xBW 变量中
junk = bwboundaries(E);%椭圆
cE  = junk{1};
yE  = cE(:,1); xE = cE(:,2);
% Vectores unitarios de BW
rBW = [xBW-xc yBW-yc];
nBW = sqrt(sum(rBW.^2,2));%到质心的距离
uBW = rBW./(repmat(nBW,1,2)+eps);% 单位向量在两个轴上的分量
% Vectores unitarios de CH
rE = [xE-xc yE-yc];
nE = sqrt(sum(rE.^2,2));%凸包边界上每个点到凸包质心的距离
uE = rE./(repmat(nE,1,2)+eps);% 单位向量在两个轴上的分量
% Distancia entre vectores unitarios
D1 = dist(uBW,uE');%n*n的矩阵
[~,ind] = min(D1,[],2);%找到 D1 每行中的最小值，并返回最小值的索引ind
% Correspondencia entre puntos de BW y puntos en CH con la orientacion
% mas proxima
mdE = cE(ind,:);
% Distancia Euclidiana
D2 = sqrt((cBW(:,1)-mdE(:,1)).^2+(cBW(:,2)-mdE(:,2)).^2);
SC = sum(D2)/Pbw.Perimeter;% 形状类别
%---------------------------------------------------------------------
% Proportinal distance，（比例距离）
S1 = bwperim(BW);
S2 = bwperim(E);
avdis1 = averagedist(S1,S2);% 计算两个边界之间的平均距离
avdis2 = averagedist(S2,S1);
PD = 100*((avdis1 + avdis2)/(2*sqrt(bwarea(E)/pi)));
%---------------------------------------------------------------------
% Features
x = [NRV OR ENC ENS LS Emax Emin R SC PD xnrl];
feats = ['sNRV','sOR','sENC','sENS','sLS','sAX_MX','sAX_MN','sROUND','sSC','sPD',fnrl];
%---------------------------------------------------------------------
function avdis = averagedist(cs,cr)
[lseg,cseg] = find(cs);
[lreal,creal] = find(cr);
[Lseg,Lreal] = meshgrid(lseg,lreal);
[Cseg,Creal] = meshgrid(cseg,creal);
dist = sqrt((Lseg-Lreal).^2+(Cseg-Creal).^2);
clear Lseg Lreal Cseg Creal
d = min(dist);
avdis = mean(d);