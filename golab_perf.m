clear;
clc;

res_dir = './results/*.txt';
spec = [res_dir(1:end-6) '.mat'];

if exist(spec, 'file') == 2
    load(spec);
    r.labsc = r.labsc(r.info(:,1) > 100,:);
    r.labsm = r.labsm(r.info(:,1) > 100,:);
    r.info = r.info(r.info(:,1) > 100,:);
else
    r = read_scores_golab(res_dir);
    save(spec, 'r');
end

% Get the anti-spoofing ROC curve
lm = r.labsm(:,1);
[am, bm, cm, dm] = perfcurve(r.info(:,6) == 0, lm, 1);
plot(am, bm);
title(['Live Classification ROC: ' spec]);
legend(['Vid Avg: ' num2str(dm)]);
saveas(gcf, [spec '_roc.jpg']);

% Get the confusion matrices
c = 1:7;
m = 0:6;

assignc = r.labsc == max(r.labsc,[],2);
assignm = r.labsm == max(r.labsm,[],2);

rc = r.info(:,4);
[cc, ac] = get_conf_mat(rc, assignc, c);
ac
cc
%cc / sum(sum(cc))

rm = r.info(:,6);
[cm, am] = get_conf_mat(rm, assignm, m);
am
cm
%cm / sum(sum(cm))

function [conf_mat, acc] = get_conf_mat(r, a, v)
    conf_mat = zeros(numel(v));
    for i = 1:numel(v)
        for j = 1:numel(v)
            conf_mat(i,j) = sum(a(:,j) .* (r == v(i)));
        end
    end
    acc = sum(sum(eye(numel(v)) .* conf_mat));
    acc = acc / sum(sum(conf_mat));
end

function result = read_scores_golab(res_dir)
    files = dir(res_dir);
    info = zeros(size(files,1), 6);
    lenc = 7;
    lenm = 7;
    labsc = zeros(size(files,1), lenc);
    labsm = zeros(size(files,1), lenm);
    for i = 1:numel(files)
        tinfo = zeros(1,6);
        tinfo(1) = str2double(files(i).name(6:8));
        tinfo(2) = str2double(files(i).name(10:11));
        tinfo(3) = str2double(files(i).name(13));
        tinfo(4) = str2double(files(i).name(15));
        tinfo(5) = str2double(files(i).name(17));
        tinfo(6) = str2double(files(i).name(19));
        info(i,1:6) = tinfo;
            
        scs = load(strcat(files(i).folder, '/', files(i).name));
        labsc(i,:) = mean(scs(:,1:lenc),1);
        labsm(i,:) = mean(scs(:,lenc+1:end),1);
    end
    result = struct();
    result.info = info;
    result.labsc = labsc;
    result.labsm = labsm;
end