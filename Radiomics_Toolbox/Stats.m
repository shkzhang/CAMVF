% 统计数据
clc;clear;
filedir = 'D:\parametric\GE参量成像肝脏造影\ICC图';
excelpath = 'D:\parametric\GE参量成像肝脏造影\HCC和ICC(GE)\ICC.xls';
[num,txt,raw] = xlsread(excelpath);
Info = [1:3 12:54];
Info(ismember(Info, [42 39])) = [];
rowN = size(raw, 1);
filedir = dir(filedir);
featn = raw(1, Info);
patientInfo = [];
patientID = [];
for k = 3:length(filedir)
    dirName = filedir(k).name;
    ind = str2double(dirName(isstrprop(dirName, 'digit')));  % 患者ID
    if isnan(ind)
        fprintf(dirName);
        return;
    end
    patientName = raw{ind, 1};
    if ~isnan(patientName)
        patientID = [patientID; cellstr(dirName(isstrprop(dirName, 'digit')))];
        patientInfo = [patientInfo; raw(ind,Info)];
    end
end
feats = ['ID' featn];
patientInfo = [patientID patientInfo];
result=[feats;patientInfo];
s=xlswrite('icc.xlsx',result);

