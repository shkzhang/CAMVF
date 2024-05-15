function [T, filenames] = select_im(filepath)
filedir=dir([filepath '/Image*.jpg']);
T = [];
filenames = cell(1, length(filedir));
for i=1:length(filedir)
    filename = filedir(i).name;   % Image32.jpg
    img = imread([filepath '/' filename]);
    if size(img, 1) ~= 819
        img = imresize(img, [819 1456]);
    end
    ocrResults = ocr(img, [1,720,127,55]);  % 将 img 图像作为输入，并根据提供的坐标 [1,720,127,55] 识别这些区域内的字符
    % 识别左下角的数字“T1:0:26”
    for k = 1:length(ocrResults.Words) % T1:   0:26
        time = ocrResults.Words(k);
        time = time{1};
        t = str2double(time(end-1:end)); % NaN    26
        if isnan(t)  % 用于拿到第二个字符串
            continue;
        end
        if strcmp(time(1),'1') % 感觉像时间换算
            t = t + 60;
        end
        if t >= 8 && t < 65 
            T = [T t];
            filenames{i} = filename;
            break;
        else
            fprintf([filepath '/' filename '\n']); % ../GE参量成像肝脏造影/ICC图/10-WANNG ZHONG YU/Image35.jpg
        end
    end  
end
if length(T) ~= length(filenames)
    warning([filepath '/' filename '\n']);
end
if length(T) > 6
    warning([filepath '/' filename '\n']);
end
% 检查 T 数组的长度是否与 filenames 数组的长度不相等,不相等，显示文件路径。
% 检查 T 数组的长度是否大于 6,如果是，显示文件路径。
[T, idx] = sort(T);
filenames = filenames(idx);
% 对 T 数组进行排序，并按相同的顺序对 filenames 数组进行重排，保持对应关系
end





