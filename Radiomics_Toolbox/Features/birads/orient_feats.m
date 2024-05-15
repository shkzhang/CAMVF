% ORIENT_FEATS Compute BI-RADS orientation features.
%   [X,FEAT] = ORIENT_FEATS(BW) computes BI-RADS orientation features according 
%   to the BI-RADS lexicon defined for breast ultrasound, where BW is the binary 
%   shape of the lesion:
%   
%   BI-RADS feature         Quantitative feature
%   ---------------         ----------------------------
%   Orientation             
%                           Angle of major axis of equivalent ellipse，等效椭圆的主轴角度
%                           Depth-to-width ratio，深度与宽度比
%   
%   Example:
%   -------
%   load('BUS01.mat');   
%   [x,feat] = orient_feats(Smanual);
%
%   See also BIRADS_FEATS BOUND_FEATS ECHO_FEATS MARGIN_FEATS SHAPE_FEATS
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
%   ORIENT_FEATS Version 1.0 (Matlab R2014a Unix)
%   December 2016
%   Copyright (c) 2016, Wilfrido Gomez Flores
% ------------------------------------------------------------------------

function [x,feats] = orient_feats(BW)
BW = double(BW);%双精度类型
Pbw = regionprops(BW,'Area','Centroid','Perimeter');
xc = Pbw.Centroid(1); yc = Pbw.Centroid(2);%质心坐标
A = Pbw.Area;%面积
[y,x] = find(BW);%病变区域
% Calcula los momentos de segundo orden del objeto binario original，原始二值形状的二阶矩
Sxx = (1/A)*sum((x-xc).^2);
Syy = (1/A)*sum((y-yc).^2);
Sxy = (1/A)*sum((x-xc).*(y-yc));
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Orientacion
theta = 0.5*atan((2*Sxy)/(Sxx-Syy));%根据二阶矩计算角度 theta，这里使用了反正切函数。
if (Sxx > Syy) && (theta > 0)%根据 Sxx、Syy 和 theta 的值，对 theta 进行修正，确保其落在适当的范围内。
    theta = 1*theta;
elseif (Sxx > Syy) && (theta < 0)
    theta = -1*theta;
elseif (Sxx < Syy) && (theta > 0)
    theta = pi/2 - theta;
elseif (Sxx < Syy) && (theta < 0)
    theta = pi/2 - (-1*theta);
end
O = theta*180/pi;%转换成以度为单位
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Depth to width ratio
[yBW,xBW] = find(BW);%病变区域
xBWmax = max(xBW); xBWmin = min(xBW);
yBWmax = max(yBW); yBWmin = min(yBW);
DWR = (yBWmax-yBWmin)/(xBWmax-xBWmin);%这里是用的矩形区域
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Features
x = [O DWR];
feats = {'oAngle','oDWR'};