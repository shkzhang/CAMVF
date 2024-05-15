clear;clc;
addpath('..');
RUN_ME_FIRST;
Class = {'ICC'};
for ind=1:length(Class)
    class = Class{ind}; % HCC,ICC
    Path = ['E:\data\liver\GE参量成像肝脏造影\' class '\'];
    Files=dir(Path);
    Dirs = zeros(length(Files)-2,1);
    for i = 3:length(Files)
        Dirs(i-2) = str2double(regexp(Files(i).name, '\d+', 'match'));%使用正则匹配获得数字
    end
    Dirs = sort(Dirs);
    Feats = [];
    for i=1:length(Dirs)
        filename1 = num2str(Dirs(i));
        if exist(['../raw/' class '/CEUS_' filename1 '.xlsx'], 'file')
            continue;
        end
        path = [Path filename1 '-*'];
        file_dir = dir(path);
        path = [Path file_dir(1).name]; % ../GE参量成像肝脏造影/HCC/2-LIU SHI MIN-HCC
        
        %% read mask
        if class == 'HCC'
            BW = imread([path '\Mask2_.png']);
        else
            files = dir(fullfile(path, '*Xnip*'));
            if ~isempty(files)
                BW = imread([path '\Mask3_.png']);
            else
                BW = imread([path '\Mask2_.png']);
            end
        end
        if numel(size(BW)) > 2
            BW = imbinarize(rgb2gray(BW));
        else
            BW = imbinarize(BW);
        end
        
        imgs_dir=dir([path '\imgs']);
        imgs_Dirs = zeros(length(imgs_dir)-2,1);
        for j = 3:length(imgs_dir)
            imgs_Dirs(j-2) = str2double(regexp(imgs_dir(j).name, '\d+', 'match'));%使用正则匹配获得数字
        end
        imgs_Dirs = sort(imgs_Dirs);
        
        for j = 1:length(imgs_Dirs)
            im_path = [num2str(imgs_Dirs(j)) '.png'];
            I  = imread([path '\imgs\' im_path]);
            H = size(I, 1);
            W = size(I, 2);
%             if abs(size(BW,1)-H) > 10
%                 disp(path);
%                 return;
%             end
            BW = imresize(BW, [H W]);
            %% extract boundary
            B = imdilate(BW,ones(10))-imerode(BW,ones(10));
            B = imbinarize(B);
            %% extract features
            feats = [];
            try
                [x, featname] = Preprocessing_and_FeatureExtraction(I,BW,B);
                feat = [imgs_Dirs(j) x];
                disp([path '\imgs\' im_path]);
            catch
                error(['Extracting features failed from ' path '/' filename1 '\n']);
                return;
            end
            feats = [feats; feat];
            Feats = [Feats; feats];
            
        end
        
        [m,p]=size(Feats);
        feat_cell=num2cell(Feats,m);
        featname = ['time' featname];
        result=[featname;feat_cell];
        % 保存文件
        save_File_Path = ['../raw/' class '/'];  % 替换为你的文件路径
        % 检查文件是否存在
        if exist(save_File_Path, 'dir') < 2 % 文件不存在，创建文件
            mkdir(save_File_Path);
        end
        xls_file = [save_File_Path 'CEUS_' num2str(Dirs(i)) '.xlsx'];
        s=xlswrite(xls_file,result);
    end
end

