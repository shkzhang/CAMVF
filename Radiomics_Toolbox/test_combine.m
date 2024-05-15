clear;clc;
addpath('..');
RUN_ME_FIRST;
ALL = ['HCC';'ICC'];
num = size(ALL);
for class=1:num(1)
    Choice = ALL(class,:); % ICC
    Path = ['../GE参量成像肝脏造影/' Choice '/'];
    Files=dir(Path);
    Dirs = zeros(length(Files)-2,1);
    for i = 3:length(Files)
        Dirs(i-2) = str2double(regexp(Files(i).name, '\d+', 'match'));%使用正则匹配获得数字
    end
    Dirs = sort(Dirs);

    Feats = [];
    for i=1:length(Dirs)
        filename1 = num2str(Dirs(i));
        % match the dir by *
        path = [Path filename1 '-*'];
        file_dir = dir(path);
        path = [Path file_dir(1).name]; % ../GE参量成像肝脏造影/HCC/2-LIU SHI MIN-HCC

        %% read image
        images = [dir([path '/屏幕截图*']) dir([path '/Xnip*']) dir([path '/图.png'])]; % 两种开头的，两种结尾的
        filename2 = images(1).name;
        flag = contains(filename2, 'Xnip'); % 0 是无需调整， 1 是需要调整
        if flag
            images = dir([path '/Image*.jpg']);
            filename2 = images(length(images)).name; % Image40.jpg
        end
        I  = imread([path '/' filename2]);

        %% read mask
        BW = imread([path '/Mask1.png']); % mask of US
        BW_2 = imread([path '/Mask2.png']); % mask of CEUS
        % mask preprocess
        if numel(size(BW)) > 2
            BW = imbinarize(rgb2gray(BW));
            BW_2 = imbinarize(rgb2gray(BW_2));
        else
            BW = imbinarize(BW);
            BW_2 = imbinarize(BW_2);
        end

        %% judge size of image == or ~=
        BW_size = size(BW);
        I_size = size(I);
        b_1 = BW_size(1);b_2 = BW_size(2);
        i_1 = I_size(1);i_2 = I_size(2);
        if (b_1 ~= i_1 || b_2 ~= i_2)
            fprintf([path '/' filename2 '--------------------------------------' '\n']); 
        end

        %% extract features
        feats = [];
        try
            [bw_x,bw_feats] = BW_Compare(BW,BW_2);
            [x, featname] = Preprocessing_and_FeatureExtraction(I,BW);
            feat = [Dirs(i) bw_x x];
            fprintf([path '/' filename2 '\n']);  % ../GE参量成像肝脏造影/HCC/2-LIU SHI MIN-HCC/屏幕截图 2021-09-23 232542.png
        catch
            fprintf(['Extracting features failed from ' path '/' filename2 '\n']);
            return;
        end
        feats = [feats; feat];
        Feats = [Feats; feats];
        
    end

    [m,p]=size(Feats);
    feat_cell=num2cell(Feats,m);
    featname = ['ID' bw_feats featname];
    result=[featname;feat_cell];
    % 保存文件
    save_File_Path = '../result';  % 替换为你的文件路径
    % 检查文件是否存在
    if exist(save_File_Path, 'dir') < 2 % 文件不存在，创建文件
        mkdir(save_File_Path);
    end
    xls_file = [save_File_Path '/' Choice '_US.xlsx'];
    s=xlswrite(xls_file,result);
end
    
%=========================================================================%
function [bw_x,bw_feats] = BW_Compare(US,CEUS)
    P_US = regionprops(US,'Area');
    P_CEUS = regionprops(CEUS,'Area');
    A_US = P_US.Area;
    A_CEUS = P_CEUS.Area;
    bw_x = A_US - A_CEUS;
    bw_feats = 'area_diff';
end