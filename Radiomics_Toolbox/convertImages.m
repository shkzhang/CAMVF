clc;clear;close all;
RUN_ME_FIRST;
Path = '../GE参量成像肝脏造影/ICC图/';
filedir=dir(Path);
Feats = [];
Patients = [];
for i=1:length(filedir)
    filename = filedir(i).name;
    if strcmp(filename,'.') || strcmp(filename,'..')
        continue;
    else
        fp = select_im([Path filename]);
        im = imread([Path filename '/' fp]);
        bw = imbinarize(imread([Path filename '/Mask3.png']));
        [im,bw] = cropI(im,bw,15);
%         imshow(labeloverlay(im,bw, 'Colormap','cool','Transparency',0.75));
        cim = convertI(im);
        save([Path filename '/im.mat'], 'cim');
%         imagesc(cim);colormap turbo;figure;imshow(im);

    end
end




