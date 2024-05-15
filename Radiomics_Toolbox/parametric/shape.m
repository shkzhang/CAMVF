function [x,feats] = shape(BW)
%SHAPE 此处显示有关此函数的摘要
%    'maxax' - Major axis length of the equivalent ellipse.
%    'minax' - Minor axis length of the equivalent ellipse.
%    'ls'    - Long axis to short axis ratio.
%     'mean'  - Mean value.
%     'std'   - Standard deviation.
%     'ar'    - Area ratio.
%     'rough' - Roughness.
[x_equ,feats_equ] = equivellipse(BW,'ls');
[x_nrl,feats_nrl] = nrl(BW,'mean','std');
x = [x_equ x_nrl];
feats = [feats_equ,feats_nrl];
end

