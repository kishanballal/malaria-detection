%% STEP 1 - Build a patch dataset from the NLM thick-smear images
% Classes: falciparum (P. falciparum parasite), vivax (P. vivax parasite),
%          uninfected (WBCs + background from uninfected patients)
%
% Output: <outDir>/<train|val|test>/<class>/*.png  plus  patches_index.csv
% Patients are split into train/val/test, so one patient never appears in
% two splits (prevents data leakage).
% Uses only base MATLAB (no Image Processing Toolbox needed).
%
% NO ROTATION (as the dataset ReadMe requires): MATLAB's imread returns the
% stored pixels and ignores the phone's EXIF orientation tag, so the
% annotation coordinates line up. The script checks every image size
% against the annotation header and skips any that do not match.

clear; clc; rng(42);

%% ---------------- settings ----------------
cfg.root            = 'C:\Users\kisha\PROJECTS\dsp\dataset';
cfg.fullData        = true;  % true = full dataset run, false = 1,300-patch sample
cfg.outDir          = 'C:\Users\kisha\PROJECTS\dsp\patches_subset';
cfg.patch           = 96;    % crop size (pixels, original resolution)
cfg.targetPerClass  = 400;   % about this many patches per class. Inf = everything
cfg.maxPerImage     = 4;     % patches taken from one image
if cfg.fullData
    cfg.outDir         = 'C:\Users\kisha\PROJECTS\dsp\patches_full';
    cfg.targetPerClass = 25000;   % balanced: about 25k patches per class (75k total)
    cfg.maxPerImage    = 30;
end
cfg.split           = [0.70 0.15 0.15];   % train / val / test by patient

%   class name     image folder                                                       annotation folder                                                    parasite label
src = {
 'falciparum', fullfile(cfg.root,'facliparum'),                                 fullfile(cfg.root,'facliparum','GT_updated'),                              "Parasite";
 'vivax',      fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_PvTk'), fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_annotations'), "Parasitized";
 'uninfected', fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Uninfected Patients'), fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Annotations'), "";
};

splits = ["train" "val" "test"];
for s = splits
    for c = 1:size(src,1)
        d = fullfile(cfg.outDir, s, src{c,1});
        if ~isfolder(d), mkdir(d); end
    end
end

h = cfg.patch/2;
idxFile = {}; idxLabel = {}; idxPatient = {}; idxSplit = {}; idxImage = {};
nSkipped = 0;

%% ---------------- main loop ----------------
for c = 1:size(src,1)
    cls = src{c,1}; imgRoot = src{c,2}; annRoot = src{c,3}; parLabel = src{c,4};

    pats = dir(imgRoot);
    pats = pats([pats.isdir] & ~startsWith({pats.name}, '.') & ~strcmp({pats.name}, 'GT_updated'));
    pats = {pats.name};
    pats = pats(randperm(numel(pats)));
    nP  = numel(pats);
    nTr = round(cfg.split(1)*nP);  nVa = round(cfg.split(2)*nP);
    splitOf = repmat("test", 1, nP);
    splitOf(1:nTr) = "train";  splitOf(nTr+1:nTr+nVa) = "val";
    quota = ceil(cfg.targetPerClass / nP);      % patches per patient
    fprintf('%s: %d patients (%d train / %d val / %d test), %g patches per patient\n', ...
            cls, nP, nTr, nVa, nP-nTr-nVa, quota);

    for p = 1:nP
        if mod(p, 10) == 0, fprintf('   %s: patient %d of %d\n', cls, p, nP); end
        imgs = [dir(fullfile(imgRoot, pats{p}, '**', '*.jpg')); ...
                dir(fullfile(imgRoot, pats{p}, '**', '*.tif*'))];
        imgs = imgs(randperm(numel(imgs)));
        nPat = 0;

        for k = 1:numel(imgs)
            if nPat >= quota, break; end
            [~, stem] = fileparts(imgs(k).name);
            annFile = fullfile(annRoot, pats{p}, [stem '.txt']);
            if ~isfile(annFile)   % some vivax patients drop the underscore in the .txt name
                annFile = fullfile(annRoot, pats{p}, [strrep(stem, '_', '') '.txt']);
            end
            if ~isfile(annFile), continue; end
            [obj, W, H] = readAnn(annFile);

            I = toRGB8(imread(fullfile(imgs(k).folder, imgs(k).name)));
            if size(I,2) ~= W || size(I,1) ~= H
                warning('Size mismatch, skipped: %s', imgs(k).name);
                nSkipped = nSkipped + 1;  continue;
            end

            n = min(cfg.maxPerImage, quota - nPat);
            labels = strings(0, 1);                 % some files list no objects
            if ~isempty(obj), labels = [obj.label]; end
            if strlength(parLabel) > 0
                par = obj(labels == parLabel);
                if isempty(par), continue; end
                par = par(randperm(numel(par), min(n, numel(par))));
                xy  = [[par.x]' [par.y]'];
            else
                wbc = obj(labels == "White_Blood_Cell");
                nW  = min(ceil(n/2), numel(wbc));
                wbc = wbc(randperm(numel(wbc), nW));
                xyW = [[wbc.x]' [wbc.y]'];
                allXY = [[obj.x]' [obj.y]'];
                xyB = sampleBackground(I, allXY, n - nW, 60);
                xy  = [xyW; xyB];
            end

            Ip = zeros(H+2*h, W+2*h, 3, 'uint8');
            Ip(h+1:h+H, h+1:h+W, :) = I;
            for j = 1:size(xy,1)
                cx = min(max(round(xy(j,1)),1), W) + h;
                cy = min(max(round(xy(j,2)),1), H) + h;
                P  = Ip(cy-h+1:cy+h, cx-h+1:cx+h, :);
                name = sprintf('%s__%s__%03d.png', pats{p}, stem, j);
                out  = fullfile(cfg.outDir, splitOf(p), cls, name);
                imwrite(P, out);
                idxFile{end+1,1} = char(out);  idxLabel{end+1,1} = cls;
                idxPatient{end+1,1} = pats{p}; idxSplit{end+1,1} = char(splitOf(p));
                idxImage{end+1,1} = imgs(k).name;
                nPat = nPat + 1;
            end
        end
    end
end

T = table(idxFile, idxLabel, idxPatient, idxSplit, idxImage, ...
          'VariableNames', {'file','label','patient','split','image'});
writetable(T, fullfile(cfg.outDir, 'patches_index.csv'));
fprintf('\nDone. %d patches written, %d images skipped for size mismatch.\n', height(T), nSkipped);
disp(groupcounts(T, {'split','label'}));

%% ---------------- optional check: annotations line up with the image ----------------
% Draws the annotations on one falciparum image. Circles should sit on parasites.
f1 = dir(fullfile(src{1,2}, '*', '*.jpg'));  f1 = f1(1);
[~, stem] = fileparts(f1.name);  [~, pdir] = fileparts(f1.folder);
[o, ~, ~] = readAnn(fullfile(src{1,3}, pdir, [stem '.txt']));
figure; imshow(imread(fullfile(f1.folder, f1.name))); hold on;
isP = [o.label] == "Parasite";
th = linspace(0, 2*pi, 40);
for q = find(isP)
    rq = max(o(q).r, 8);
    plot(o(q).x + rq*cos(th), o(q).y + rq*sin(th), 'y-', 'LineWidth', 1.5);
end
plot([o(~isP).x], [o(~isP).y], 'c+', 'MarkerSize', 12, 'LineWidth', 2);
title('Yellow = parasite, cyan = WBC');

%% ================= helper functions =================
function [obj, W, H] = readAnn(f)
% NLM format. Line 1: count,width,height
% Other lines: id,label,comment,shape,nPoints,x1,y1[,x2,y2,...]
% Circle = centre point + a point on the edge. Point = one point.
    L = strtrim(splitlines(string(fileread(f))));
    L(L == "") = [];
    hdr = str2double(split(L(1), ','));
    W = hdr(end-1);  H = hdr(end);
    obj = struct('label', {}, 'x', {}, 'y', {}, 'r', {});
    for i = 2:numel(L)
        t = split(L(i), ',');
        if numel(t) < 7, continue; end
        v  = str2double(t(6:end));
        xs = v(1:2:end);  ys = v(2:2:end);
        switch t(4)
            case "Circle"
                x = xs(1); y = ys(1); r = hypot(xs(2)-xs(1), ys(2)-ys(1));
            case "Point"
                x = xs(1); y = ys(1); r = NaN;
            otherwise     % Polygon
                x = mean(xs); y = mean(ys); r = max(hypot(xs-x, ys-y));
        end
        obj(end+1) = struct('label', t(2), 'x', x, 'y', y, 'r', r); %#ok<AGROW>
    end
end

function pts = sampleBackground(I, avoidXY, n, minDist)
% Random points inside the round field of view, away from annotated objects.
    pts = zeros(0, 2);
    if n <= 0, return; end
    % field-of-view mask on a 4x smaller grey image (no toolbox needed)
    g   = mean(double(I(1:4:end, 1:4:end, :)), 3);
    fov = g > 25;
    [h4, w4] = size(fov);
    H = size(I,1); W = size(I,2);
    ang = (0:7) * pi/4;  rr = 20;    % ring of 8 points, 80 px away at full size
    for tries = 1:1000
        if size(pts,1) >= n, break; end
        x = randi(W); y = randi(H);
        cy = round(y/4); cx = round(x/4);
        ry = round(cy + rr*sin(ang)); rx = round(cx + rr*cos(ang));
        if any(ry < 1 | ry > h4 | rx < 1 | rx > w4), continue; end
        if ~all(fov(sub2ind([h4 w4], [ry cy], [rx cx]))), continue; end
        if ~isempty(avoidXY) && min(hypot(avoidXY(:,1)-x, avoidXY(:,2)-y)) < minDist, continue; end
        pts(end+1, :) = [x y]; %#ok<AGROW>
    end
end

function I = toRGB8(I)
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    if size(I,3) > 3,  I = I(:,:,1:3); end
    if isa(I, 'uint16')
        I = uint8(double(I) / 257);
    elseif isfloat(I)
        I = uint8(255 * I);
    elseif islogical(I)
        I = uint8(255 * I);
    end
end
