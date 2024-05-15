function [x,feats] = FeatureExtraction(im, bw, T)
% im 是时间分布图像, bw 是裁剪后的二值图像
[x_firstorder,feats_firstorder] = firstorder(im, bw); % 提取到的统计特征,35+8
% 初始灌注区域
[x_init,feats_init] = InitialPerfusion(im, bw, T);  % 获得前两个时刻的面积比率信息，包含（最小时间-2），T(1)时间面积比，T(1)+T(2)面积比，是否超过0.55,  4
% 无灌注区域
[x_Np,feats_Np] = NonPerfusion(im, bw);  % 无灌注区域，无灌注区域的面积比率,2
x = [x_firstorder x_init x_Np];
feats = [feats_firstorder, feats_init, feats_Np];  % 特征合并
end

