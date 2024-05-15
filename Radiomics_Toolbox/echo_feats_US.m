% the echo of US
function [x,feats] = echo_feats_US(I,BW)
% Internal pattern
I = double(I);
J = I(BW)+1;
EPi = mean(J);  % Average internal mass intensity
N = sum(BW(:));
H = accumarray(J,ones(N,1),[256 1],@sum,0);
p = 0.25*N;
for i = 256:-1:1
    if sum(H(i:256)) > p
        break;
    end
end
EPc = (mean(J(J>=i)) - EPi)/EPi; % Intensity difference between the 25 % 
                                 % brighter pixels and whole tumor pixels

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Intensity autocorrelation
xa = autocorr(I,BW);
% GLCM  Contrast, Correlation, Energy, Homogeneity
D = [1 2 4 8]; % Distances in pixels
[xglcm,fglcm] = glcm(I,BW,64,D,1,'mean','contr','corrm','energ','homom');
% Kurtosis
[xlaw,flaw] = lawsenergy(I,BW,1,'kur');
% histogram features
[xhis,fhis] = histfeatures(I,BW,'bag','ag','ahg','sd','egy','diff');  
% local binary pattern variance
[xlbp,flbp] = lbpv(I,BW);
% Features
x = [EPi EPc xa xglcm xlaw xhis xlbp];
feats = ['eEPi','eEPc','eACOR',fglcm, flaw, fhis, flbp];
end

