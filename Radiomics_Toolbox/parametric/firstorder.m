function [x,feats] = firstorder(PI, bw)
%UNTITLED5 ����߽硢���ĺ������ATͳ�Ʋ���
%   �˴���ʾ��ϸ˵��
% PI ��ʱ��ֲ�ͼ��, bw �ǲü���Ķ�ֵͼ��
[x_int, feat_int] = textureinterior(PI,bw); % ��bw����ȡ 5�֣� ÿ��������7������
US = double(PI);
Ig = US(logical(bw));  % ��PIͼ������ȡ
% Ig(isnan(Ig))=[];
Ef = mode(Ig);
Em = mean(Ig);
Emd = median(Ig);
Etd = std(Ig);
% Eskw = skewness(Ig);
% Ek = kurtosis(Ig);
Eq = entropy(uint8(Ig));
Eq1 = quantile(Ig, 0.25); % �ķ�λ�� Q1
Eq3 = quantile(Ig, 0.75); % �ķ�λ�� Q3
Ei = Eq3 - Eq1;  % ����ͼ���

x = [x_int  Ef Em Emd Etd Eq Eq1 Eq3 Ei];
feats = [feat_int,'freq','mean','median','std','entropy','quantile1','quantile3','quantileI'];
end

