% SHAPE_FEATS_US Compute lesion shape features.
%   [X,FEAT] = SHAPE_FEATS_US(BW) computes shape features, where BW is the binary shape of the lesion
%   
%   BI-RADS feature         Quantitative feature
%   ---------------         -----------------------------------------
%   Shape                   
%                           Normalized residual value   (NRV)
%                           Normalized radial lenghth    (NRL)
%                           Overlap with equivalent ellipse (EE)                
%                           Long axis to short axis ratio of EE   
%                           Compactness or roundness
%                           Major and minor axis length of EE
%                           Fractal dimension
%                           Spiculation 
%
%   Example:
%   -------
%   load('BUS01.mat');   
%   [x,feat] = shape_feats(Smanual);
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

function [x,feats] = shape_feats_US(I,BW)
% Crop
BW = bwareaopen(BW,30);% 移除小的区域
[y,x] = find(BW);
xmn = min(x);
xmx = max(x);
ymn = min(y);
ymx = max(y);
BW2 = BW(ymn:ymx,xmn:xmx);
BW2 = padarray(BW2,[1 1],0,'both');
%---------------------------------------------------------------------
%% Normalized Residual Value
BW_props = regionprops(BW2,'ConvexHull','Perimeter','Centroid','Area');
CH = roipoly(BW2,BW_props.ConvexHull(:,1),BW_props.ConvexHull(:,2));
NRV = bwarea(xor(BW2,CH))/bwarea(CH); 
%---------------------------------------------------------------------
%% Normalized radial lenghth 
[xnrl,fnrl] = nrl(BW);
%---------------------------------------------------------------------
%% Elipse equivalente
Pbw = regionprops(BW,'Area','Centroid','Perimeter');
A = Pbw.Area;
xc = Pbw.Centroid(1); yc = Pbw.Centroid(2);
[y,x] = find(BW);
[xx,yy] = meshgrid(1:size(BW,2),1:size(BW,1)); 
Sxx = (1/A)*sum((x-xc).^2);
Syy = (1/A)*sum((y-yc).^2);
Sxy = (1/A)*sum((x-xc).*(y-yc)); 
coef = (1/(4*Sxx*Syy - Sxy^2))*[Syy -Sxy;-Sxy Sxx];
a = coef(1,1); b = coef(1,2); c = coef(2,2); 
E = (a*(xx-xc).^2 + 2*b*(xx-xc).*(yy-yc) + c*(yy-yc).^2) < 1; 
E_props = regionprops(E,'Perimeter','MinorAxisLength','MajorAxisLength');

%% Overlap with equivalent ellipse
OR  = bwarea(BW&E)/bwarea(BW|E);	
%% Long axis to short axis ratio of EE
LS = E_props.MajorAxisLength/E_props.MinorAxisLength;
%% Major and minor axis length of EE
Emax = E_props.MajorAxisLength; 
Emin = E_props.MinorAxisLength;
%% Compactness
R = 1 - ((4*pi*A)/(Pbw.Perimeter^2)); 
%% fractal feature
[xfra,ffra] = fractaltexture(I,BW);
xfra = xfra(1);
ffra = ffra(1);
%% Spiculation feature | Number of skeleton end-points
[xspi,fspi] = spiculation(BW,'spic');
%% Features
x = [NRV xnrl OR LS Emax Emin R xfra xspi];
feats = ['sNRV',fnrl,'sOR','sLS','sAX_MX','sAX_MN','sROUND',ffra,fspi];
%---------------------------------------------------------------------


