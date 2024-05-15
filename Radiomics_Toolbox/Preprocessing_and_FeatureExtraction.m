function [x,feats] = Preprocessing_and_FeatureExtraction(I,BW,B)
%% imageͨ��ת��
if numel(size(I))>2
    I1 = rgb2gray(I);
else
    I1 = I;
end
%% ͼ��Աȶ���ǿ
% I2 = fuzzyequ(I1);

%% ͼ���˲�ȥ��
% I3  = uint8(isf(I1,5));
I3 = I1;
% [c,s] = wavedec2(I1,2,'sym4');
% a1 = wrcoef2('a',c,s,'sym4');
% I3 = uint8(a1);
% figure;
% subplot 121; imshow(I); title('Original Image');
% subplot 122; imshow(mat2gray(I3)); title('Filtered Image');

%% ��״����������ȡ
if nargin > 2
    [x,feats] = Custom_feats_CEUS(I3,BW,B);
else
    [x,feats] = Custom_feats_US(I3,BW);
    % [x,feats] = Custom_feats_ELA(I3,BW);
end

end

