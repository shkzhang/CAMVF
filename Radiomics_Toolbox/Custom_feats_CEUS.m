%%   feature                 Quantitative feature
%   ---------------         ----------------------------
%   Epoch_pattern           from lesion as well as internal, bounary         
 
function [x,feats] = Custom_feats_CEUS(I,BW,B)

%%  epoch pattern of lesion 
[x_R,feats_R] = echo_feats_US(I,BW);  % 119-dim

%%  Kurtosis and histogram features of lesion boundary
% B = imdilate(BW,ones(10))-imerode(BW,ones(10));
% B = imbinarize(B);
[xlaw,flaw] = lawsenergy(I,B,1,'kur');
[xhis,fhis] = histfeatures(I,B,'bag','ag','ahg','sd','egy','diff');  
x_B = [xlaw xhis];
feats_B = [flaw, fhis];

%% histogram feature of lesion internal
C = imerode(BW,ones(20));
[x_C,feats_C] = histfeatures(I,C,'ag','ahg','sd','egy'); 

%% Relative brightness of the tumor and adjacent tissue
Diff_EGY_RB = x_R(strcmp(feats_R, 'hEgy' ))-x_B(strcmp(feats_B, 'hEgy' ));

%% Relative brightness of the tumor and internal tissue
Diff_AG_RC = x_R(strcmp(feats_R, 'hAG' ))-x_C(strcmp(feats_C, 'hAG' ));
Diff_EGY_RC = x_R(strcmp(feats_R, 'hEgy' ))-x_C(strcmp(feats_C, 'hEgy' ));

%% rename feature of boundary/internal
for i = 1:length(feats_B)
    feats_B(i) = strcat(feats_B(i),'_B');
end
for i = 1:length(feats_C)
    feats_C(i) = strcat(feats_C(i),'_C');
end

%% Minimum side difference
[xpab,fpab] = pab(I,BW,'msd'); 
%% normalized radial gradient
[xboun,fboun] = bound_feats(I,BW); 

x = [x_R x_B x_C Diff_EGY_RB Diff_AG_RC Diff_EGY_RC xpab xboun];
feats = [feats_R, feats_B, feats_C, 'Diff_EGY_RB', 'Diff_AG_RC', 'Diff_EGY_RC', fpab, fboun];
end

