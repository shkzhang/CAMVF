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
        path = [Path filename1 '-*'];
        file_dir = dir(path);
        path = [Path file_dir(1).name]; % ../GE参量成像肝脏造影/HCC/2-LIU SHI MIN-HCC
        %% read image
        if class == 'HCC'
            BW = imread([path '\Mask2_.png']);
            I  = imread([path '\peak.png']);
        else
            files = dir(fullfile(path, '*Xnip*'));
            if ~isempty(files)
                BW = imread([path '\Mask3_.png']);
                I  = imread([path '\peak.png']);
            else
                I  = imread([path '\peak.png']);
                BW = imread([path '\Mask2_.png']);
            end
        end
        
        H = size(I, 1);
        W = size(I, 2);
        
        if numel(size(BW)) > 2
            BW = imbinarize(rgb2gray(BW));
        else
            BW = imbinarize(BW);
        end
        %% match mask and image
        if (size(BW, 1) ~= H || size(BW, 2) ~= W)
            error([path '/' filename1 '--------------------------------------' '\n']);
        end
        %% extract boundary
        B = imdilate(BW,ones(10))-imerode(BW,ones(10));
        B = imbinarize(B);
        
        files = dir(fullfile(path, 'plot_*'));
        pattern = '(\d+_\d+_\d+)';
        matches = regexp(files.name, pattern, 'tokens');
        recognizedText = matches{1}{1};
        parts = split(recognizedText, {'_'});
        start = str2double(parts{1});
        peak = str2double(parts{2});
        ending = str2double(parts{3});
        %% extract features
        feats = [];
        try
            [x, featname] = Preprocessing_and_FeatureExtraction(I,BW,B);
            feat = [Dirs(i) start peak ending x];
            disp([path '\' filename1]);
        catch
            error(['Extracting features failed from ' path '/' filename1 '\n']);
            return;
        end
        feats = [feats; feat];
        Feats = [Feats; feats];
        
    end
    
    [m,p]=size(Feats);
    feat_cell=num2cell(Feats,m);
    featname = ['ID' 'start' 'peak' 'ending' featname];
    result=[featname;feat_cell];
    % 保存文件
    save_File_Path = '../raw';  % 替换为你的文件路径
    % 检查文件是否存在
    if exist(save_File_Path, 'dir') < 2 % 文件不存在，创建文件
        mkdir(save_File_Path);
    end
    xls_file = [save_File_Path '/' class '_CEUS.xlsx'];
    s=xlswrite(xls_file,result);
end