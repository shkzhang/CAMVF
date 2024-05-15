clear;clc;
RUN_ME_FIRST;
filepath = 'E:\data\liver\GE参量成像肝脏造影\HCC\';
dirs = dir(filepath);
Dirs = zeros(length(dirs)-2,1);
for i = 3:length(dirs)
    Dirs(i-2) = str2double(regexp(dirs(i).name, '\d+', 'match'));%使用正则匹配获得数字
end
Dirs = sort(Dirs);
Feats = [];
filenames = {};
for i=1:length(Dirs)
    filename1 = num2str(Dirs(i));
    path = [filepath filename1 '-*'];
    file_dir = dir(path);
    folder_path = [filepath file_dir(1).name];
    ID = filename1;
    files = dir(fullfile(folder_path, '*Xnip*'));
    if ~isempty(files)
        BW_US = imread([folder_path '\Mask1.png']);
        files = dir(fullfile(folder_path, 'Image*.jpg'));
        I = imread([folder_path '\' files(end).name]);
    else
        I  = imread([folder_path '\peak.png']);
        BW_US = imread([folder_path '\Mask1_.png']);
    end
 
    H = size(I, 1);
    W = size(I, 2);
    % mask preprocess
    if (numel(size(BW_US)) > 2)
        BW_US = imbinarize(rgb2gray(BW_US));
    else
        BW_US = imbinarize(BW_US);
    end
    if (size(BW_US, 1) ~= H || size(BW_US, 2) ~= W)
        error([folder_path '/' image_path '--------------------------------------' '\n']);
    end
    
    %% extract features
    feats = [];
    try
        [x, featname] = Preprocessing_and_FeatureExtraction(I,BW_US);
        disp(file_dir(1).name);  % ../GE参量成像肝脏造影/HCC/2-LIU SHI MIN-HCC/屏幕截图 2021-09-23 232542.png
    catch
        error(['Extracting features failed from ' filename '\n']);
        return;
    end
    feats = [feats; x];
    Feats = [Feats; feats];
    filenames{end+1} = ID;
end
[m,p]=size(Feats);
filenames = reshape(filenames,[],1);
feat_cell=num2cell(Feats,m);
feat_cell = [filenames,feat_cell];
featname = ['ID' featname];
result=[featname;feat_cell];
% 保存文件
save_File_Path = '../raw';  % 替换为你的文件路径
% 检查文件是否存在
if exist(save_File_Path, 'dir') < 2 % 文件不存在，创建文件
    mkdir(save_File_Path);
end
xls_file = [save_File_Path '/' 'HCC_US.xlsx'];
s=xlswrite(xls_file,result);