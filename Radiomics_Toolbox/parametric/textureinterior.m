function [x,feats] = textureinterior(US, BW)
% �Բ�������Ϊԭ�㣬ȡ��ͬ�뾶�������ͼ��ɫ�ֲ��������໥֮��Ĳ�ֵ���뾶��С��Բ����
% boundary feature
% most frequent value
% mean AT
% median AT
% standard deviation AT
% skewness and kurtosis ƫ�Ⱥͷ��
% extract boundary regions

% US ��ʱ��ֲ�ͼ��, BW�ǲü���Ķ�ֵͼ��
US = double(US);
sz=size(BW);  % [195,255]
% �����ֵͼ������ͨ��������ԣ����а���������������������
Pbw = regionprops(BW,'Area','Centroid');  % Area 28550��Pbw.Centroid [126.7789,90.0192]
xc = Pbw.Centroid(1); yc = Pbw.Centroid(2);% [126.7789,90.0192]

% ��������������� xgrid �� ygrid�����ڱ�ʾͼ��ĺ�������
[xgrid, ygrid] = meshgrid(1:sz(2), 1:sz(1));  % size 195x255 double
x = xgrid - xc;  % 195x255 double
y = ygrid - yc;  % 195x255 double
r3= sqrt(Pbw.Area / pi);  % 95.329676651854970
r0 = r3 / 4;
r1 = r3 / 2;
r2 = r3 / 4 * 3;

C0 = x.^2 + y.^2 <= r0.^2; % 1/4 r3
C1 = x.^2 + y.^2 <= r1.^2; % 1/2 r3
C2 = x.^2 + y.^2 <= r2.^2; % 3/4 r3
C3 = x.^2 + y.^2 <= r3.^2; % 1   r3
C3 = C3 - C2;% 7/16 
C2 = C2 - C1;% 5/16
C1 = C1 - C0;% 3/16
% map = double(BW);
% map(logical(C0)) = 1;
% map(logical(C1)) = 2;
% map(logical(C2)) = 3;
% map(logical(C3)) = 4;
% imagesc(map);

[x0, ~] = statsfeats(US, C0); % ����ͳ��������������������λ������׼��ķ�λ��
[x1, ~] = statsfeats(US, C1);
[x2, ~] = statsfeats(US, C2);
[x3, featname] = statsfeats(US, C3);
x03 = x3 - x0;
feats = cell(1);
feats_suff = {'C0','C1','C2','C3','C_diff'};
for i = 1:length(feats_suff)
    for j = 1:7 % ���������ͳ������
        feats{(i-1)*7+j} = [featname{j} '_' feats_suff{i}];
    end
end
x = [x0 x1 x2 x3 x03];
end

