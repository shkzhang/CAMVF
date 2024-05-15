Path = '../GE参量成像肝脏造影/ICC图/';
filedir=dir(Path); % 
Feats = [];
Patients = [];
for i=1:length(filedir)
    filename = filedir(i).name;
    if strcmp(filename,'.') || strcmp(filename,'..') % 排除两种特殊情况
        continue;
    else
        [T, filenames] = select_im([Path filename]);
        disp([Path filename]);
        disp([Path filename '/Mask3.png']);
        break;
        imread([Path filename '/Mask3.png']);
        disp(filenames);
        disp('=================================');
    end
end