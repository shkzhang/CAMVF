clear;clc;
addpath('..');
RUN_ME_FIRST;
Class = {'HCC', 'ICC'};
for ind=1:length(Class)
    class = Class{ind};
    Path = ['../../data/' class '/'];
    Files=dir(Path);
    Dirs = zeros(length(Files)-2,1);
    for i = 3:length(Files)
        Dirs(i-2) = str2double(regexp(Files(i).name, '\d+', 'match'));
    end
    Dirs = sort(Dirs);
    Feats = [];
    Patients = [];
    for i=1:length(Dirs)
        filename1 = num2str(Dirs(i));
        path = [Path filename1 '-*'];
        file_dir = dir(path);
        dirname = file_dir(1).name; 
        [T, filenames] = select_im([Path dirname]);  
        im = imread([Path dirname '/' filenames{1}]);
        bw = imbinarize(imread([Path dirname '/Mask3.png']));
        if (size(bw,1)~=size(im, 1) || size(bw,2)~=size(im, 2))
            bw = imbinarize(imread([Path dirname '/Mask2.png']));
        end 
        rect = cropI(bw,15);  
        bw = imcrop(bw,rect); 
        cim = convertI(T, filenames, rect, [Path dirname]); 
        % %         save([Path filename '/im.mat'], 'cim');
        % %         imagesc(cim);colormap turbo;figure;imshow(im);
        try
            [feat, featn] = FeatureExtraction(cim, bw, T);% ���� �� ����������
            fprintf([Path dirname '\n']);
        catch ErrorInfo
            error(['Extracting shape and texture features from ' dirname '\n']);
        end
        Feats = [Feats; feat];
        Patients = [Patients; cellstr(dirname(isstrprop(dirname,'digit')))];
    end

    [m,p]=size(Feats); % ���� �� ��������
    feat_cell= num2cell(Feats,p);
    feat_cell = [Patients feat_cell];
    feats = ['ID' featn];
    result=[feats;feat_cell];
    % �����ļ�
    save_File_Path = '../raw';  % �滻Ϊ����ļ�·��
    % ����ļ��Ƿ����
    if exist(save_File_Path, 'dir') < 2 % �ļ������ڣ������ļ�
        mkdir(save_File_Path);
    end
    xls_file = [save_File_Path '/' class '_PARA.xlsx'];
    s=xlswrite(xls_file,result);
end

