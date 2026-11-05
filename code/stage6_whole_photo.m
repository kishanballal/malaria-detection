%% STAGE 6 - WHOLE-PHOTO DETECTION AND DIAGNOSIS (demo)
% Uses the trained M3 (EfficientNet-b0) on complete 4032 x 3024 smear photos,
% not on the annotated patches:
%   1. find candidate objects automatically (dark stained blobs)
%   2. cut a 96 x 96 patch around each candidate, Reinhard-normalise it
%      with the same training reference, classify it with M3
%   3. photo diagnosis: "uninfected" if fewer than T parasites are found,
%      otherwise the species with the larger total parasite score
%   4. patient diagnosis: majority vote of that patient's photos
% The threshold T is chosen on VALIDATION patients, then applied unchanged
% to TEST patients. Expert annotations are used only to score the detection.
% No rotation anywhere.

clear; clc; rng(42);

cfg.root      = 'C:\Users\kisha\PROJECTS\dsp\dataset';
cfg.indexCsv  = 'C:\Users\kisha\PROJECTS\dsp\patches_full\patches_index.csv';
cfg.modelFile = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full\M3_efficientnetb0_finetuned.mat';
cfg.refFile   = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing_full\stain_reference.mat';
cfg.outDir    = 'C:\Users\kisha\PROJECTS\dsp\stage6_whole_photo_full';
cfg.photosPerPatient = 3;     % photos per patient used (more = slower, ~3-5 s per photo)
cfg.patch     = 96;
cfg.contrast  = 20;           % how much darker than its surroundings a blob must be (grey levels)
cfg.minArea   = 8;            % blob area limits at half resolution (pixels)
cfg.maxArea   = 6000;
cfg.maxCand   = 1500;         % at most this many candidates per photo (darkest first)
cfg.minScore  = 0.90;         % a candidate counts as a parasite only if M3 is this sure
cfg.matchDist = 30;           % px: detection counts as matching an annotated parasite
cfg.batch     = 64;
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

src = {
 'falciparum', fullfile(cfg.root,'facliparum'),                                 fullfile(cfg.root,'facliparum','GT_updated'),                              "Parasite";
 'vivax',      fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_PvTk'), fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_annotations'), "Parasitized";
 'uninfected', fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Uninfected Patients'), fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Annotations'), "";
};

L = load(cfg.modelFile);  net = L.M3;  classes = L.classes(:)';
inSz = net.Layers(1).InputSize(1:2);
R = load(cfg.refFile);  ref = R.ref;
iPar = find(ismember(classes, {'falciparum','vivax'}));

%% ---------------- choose photos: same patient split as training ----------------
Lx = strtrim(readlines(cfg.indexCsv));  Lx(Lx == "") = [];
P  = split(Lx(2:end), ",");            % file,label,patient,split,image
T  = unique(table(P(:,2), P(:,3), P(:,4), P(:,5), 'VariableNames', {'cls','patient','split','image'}));
sel = table();
for s = ["val" "test"]
    pts = unique(T.patient(T.split == s));
    for k = 1:numel(pts)
        r = find(T.split == s & T.patient == pts(k));
        r = r(randperm(numel(r), min(cfg.photosPerPatient, numel(r))));
        sel = [sel; T(r, :)]; %#ok<AGROW>
    end
end
fprintf('Photos selected: %d validation, %d test\n', sum(sel.split=="val"), sum(sel.split=="test"));

%% ---------------- run detection + classification on every selected photo ----------------
n = height(sel);
res = table('Size', [n 8], 'VariableTypes', {'string','string','string','string','double','double','double','double'}, ...
    'VariableNames', {'split','patient','image','trueClass','nCandidates','nFalciparum','nVivax','nAnnotated'});
res.split = sel.split; res.patient = sel.patient; res.image = sel.image; res.trueClass = sel.cls;
sumF = zeros(n,1); sumV = zeros(n,1); nMatched = zeros(n,1); nParTrue = zeros(n,1);
keep = cell(n,1);       % detections, kept for the overlay figures
tStart = tic;
for i = 1:n
    c = find(strcmp(src(:,1), sel.cls(i)));
    [imgPath, annPath] = findFiles(src(c,:), sel.patient(i), sel.image(i));
    I = toRGB8(imread(imgPath));
    xy = findCandidates(I, cfg);
    X = cutPatches(I, xy, cfg.patch, ref);
    if isempty(xy)
        sc = zeros(0, numel(classes));
    else
        sc = minibatchpredict(net, augmentedImageDatastore(inSz, X), 'MiniBatchSize', cfg.batch);
    end
    [mx, lab] = max(sc, [], 2);
    isPar = ismember(lab, iPar) & mx >= cfg.minScore;
    res.nCandidates(i) = size(xy,1);
    res.nFalciparum(i) = sum(isPar & lab == find(strcmp(classes,'falciparum')));
    res.nVivax(i)      = sum(isPar & lab == find(strcmp(classes,'vivax')));
    sumF(i) = sum(sc(isPar, strcmp(classes,'falciparum')));
    sumV(i) = sum(sc(isPar, strcmp(classes,'vivax')));
    % compare with expert annotations (only for scoring)
    annXY = zeros(0,2);
    if strlength(annPath) > 0 && strlength(src{c,4}) > 0
        obj = readAnn(annPath);
        if ~isempty(obj), o = obj([obj.label] == src{c,4}); annXY = [[o.x]' [o.y]']; end
    end
    res.nAnnotated(i) = size(annXY,1);
    if ~isempty(annXY) && any(isPar)
        d = pdist2(annXY, xy(isPar,:));
        nMatched(i) = sum(min(d, [], 2) <= cfg.matchDist);
        nParTrue(i) = sum(min(d, [], 1) <= cfg.matchDist);
    end
    keep{i} = struct('xy', xy, 'lab', lab, 'isPar', isPar, 'annXY', annXY, 'path', imgPath);
    el = toc(tStart);
    fprintf('  %4d / %d  %-5s %-10s %-12s  candidates %4d  parasites F %3d V %3d (annotated %3d)  | %4.1f min left\n', ...
        i, n, sel.split(i), sel.cls(i), sel.patient(i), size(xy,1), res.nFalciparum(i), res.nVivax(i), ...
        res.nAnnotated(i), el/i*(n-i)/60);
end

%% ---------------- choose the threshold T on VALIDATION photos ----------------
nPar = res.nFalciparum + res.nVivax;
species = strings(n,1);  species(sumF >= sumV) = "falciparum";  species(sumF < sumV) = "vivax";
isVal = res.split == "val";
Ts = 1:40;  accT = zeros(size(Ts));
for k = 1:numel(Ts)
    pred = species;  pred(nPar < Ts(k)) = "uninfected";
    accT(k) = mean(pred(isVal) == res.trueClass(isVal));
end
[bestAcc, kb] = max(accT);  Tbest = Ts(kb);
fprintf('\nThreshold chosen on validation photos: T = %d parasites (validation photo accuracy %.1f%%)\n', Tbest, 100*bestAcc);
res.predicted = species;  res.predicted(nPar < Tbest) = "uninfected";
writetable(res, fullfile(cfg.outDir, 'photo_results.csv'));

%% ---------------- results on TEST photos and patients ----------------
isTe = res.split == "test";
yt = categorical(res.trueClass(isTe), classes);  yp = categorical(res.predicted(isTe), classes);
[pts, ~, g] = unique(res.patient(isTe));
pt = splitapply(@(x) mode(x), yt, g);  pp = splitapply(@(x) mode(x), yp, g);
detRecall = sum(nMatched(isTe)) / max(sum(res.nAnnotated(isTe)), 1);
parInf = isTe & res.trueClass ~= "uninfected";
detPrec = sum(nParTrue(parInf)) / max(sum(nPar(parInf)), 1);
fpUninf = mean(nPar(isTe & res.trueClass == "uninfected"));

fprintf('\n===== WHOLE-PHOTO RESULTS ON TEST PATIENTS =====\n');
fprintf('Photo-level accuracy  : %.1f%%  (%d of %d photos)\n', 100*mean(yp == yt), sum(yp == yt), numel(yt));
fprintf('Patient-level accuracy: %.1f%%  (%d of %d patients, majority vote of photos)\n', 100*mean(pp == pt), sum(pp == pt), numel(pts));
fprintf('Parasite detection    : %.1f%% of annotated parasites found (recall)\n', 100*detRecall);
fprintf('                        %.1f%% of detected parasites match an annotation (precision, infected photos)\n', 100*detPrec);
fprintf('False parasites on uninfected photos: %.1f per photo on average\n', fpUninf);
summary = table(["Photos"; "Patients"; "Parasite recall"; "Parasite precision"; "False parasites per uninfected photo"; "Threshold T (from validation)"], ...
    [100*mean(yp == yt); 100*mean(pp == pt); 100*detRecall; 100*detPrec; fpUninf; Tbest], ...
    'VariableNames', {'Measure','Value'});
writetable(summary, fullfile(cfg.outDir, 'whole_photo_summary.csv'));

% confusion matrices (photo and patient level)
for lev = 1:2
    fig = figure('Position', [50 50 620 480], 'Color', 'w');
    try, theme(fig, 'light'); catch, end
    if lev == 1, a = yt; b = yp; ttl = "Whole photos (test)"; f = 'fig_confusion_photos.png';
    else,        a = pt; b = pp; ttl = "Patients, majority vote of photos (test)"; f = 'fig_confusion_patients.png'; end
    cc = confusionchart(fig, a, b, 'Title', ttl, 'RowSummary', 'row-normalized', ...
        'DiagonalColor', [0.25 0.25 0.25], 'OffDiagonalColor', [0.65 0.65 0.65]);
    cc.FontName = 'Times New Roman';
    try, exportgraphics(cc, fullfile(cfg.outDir, f), 'Resolution', 200); catch, saveas(fig, fullfile(cfg.outDir, f)); end
end

%% ---------------- overlay figure: one test photo per class ----------------
fig = figure('Position', [30 30 1600 480], 'Color', 'w');
try, theme(fig, 'light'); catch, end
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for c = 1:3
    r = find(isTe & res.trueClass == src{c,1});
    if isempty(r), continue; end
    [~, k] = max(res.nAnnotated(r) + nPar(r));   r = r(k);   % a photo with plenty to show
    K = keep{r};  I = toRGB8(imread(K.path));
    ax = nexttile(tl);  imshow(I, 'Parent', ax);  hold(ax, 'on');
    if ~isempty(K.annXY)
        plot(ax, K.annXY(:,1), K.annXY(:,2), 'w+', 'MarkerSize', 6, 'LineWidth', 1);
    end
    f = K.isPar & strcmp(classes(K.lab)', 'falciparum');  v = K.isPar & strcmp(classes(K.lab)', 'vivax');
    plot(ax, K.xy(f,1), K.xy(f,2), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
    plot(ax, K.xy(v,1), K.xy(v,2), 'ys', 'MarkerSize', 12, 'LineWidth', 1.5);
    hold(ax, 'off');
    title(ax, sprintf('%s photo, predicted: %s\n%d falciparum, %d vivax detections, %d annotated', ...
        src{c,1}, res.predicted(r), res.nFalciparum(r), res.nVivax(r), res.nAnnotated(r)), 'Color', 'k', 'FontSize', 10);
end
title(tl, 'Red circle = falciparum, yellow square = vivax, white + = expert annotation', 'FontSize', 10);
exportgraphics(fig, fullfile(cfg.outDir, 'fig_whole_photo_examples.png'), 'Resolution', 150);

% threshold curve
fig = figure('Position', [50 50 600 420], 'Color', 'w');
try, theme(fig, 'light'); catch, end
ax = axes(fig);  plot(ax, Ts, 100*accT, 'k-o', 'MarkerSize', 3, 'LineWidth', 1.2);
xline(ax, Tbest, 'k:', sprintf('T = %d', Tbest));  grid(ax, 'on');
xlabel(ax, 'Minimum parasites to call a photo infected (T)');  ylabel(ax, 'Validation photo accuracy (%)');
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
exportgraphics(fig, fullfile(cfg.outDir, 'fig_threshold_validation.png'), 'Resolution', 200);

fprintf('Stage 6 done. Results saved in %s\n', cfg.outDir);

%% ======================= helpers =======================
function xy = findCandidates(I, cfg)
% dark stained blobs (parasites, WBC nuclei) inside the round field of view
    g  = im2single(rgb2gray(imresize(I, 0.5))) * 255;
    fov = imerode(imfill(g > 25, 'holes'), strel('disk', 15));
    bg = imgaussfilt(g, 25);                    % local background brightness
    bw = (bg - g) > cfg.contrast & fov;
    bw = bwareafilt(bw, [cfg.minArea cfg.maxArea]);
    st = regionprops(bw, g, 'Centroid', 'MeanIntensity');
    if isempty(st), xy = zeros(0,2); return; end
    xy = 2 * cat(1, st.Centroid);
    [~, o] = sort([st.MeanIntensity]);          % darkest first
    xy = xy(o(1:min(cfg.maxCand, numel(o))), :);
end

function X = cutPatches(I, xy, ps, ref)
    h = ps/2;  [H, W, ~] = size(I);
    Ip = zeros(H+ps, W+ps, 3, 'uint8');  Ip(h+1:h+H, h+1:h+W, :) = I;
    X = zeros(ps, ps, 3, size(xy,1), 'uint8');
    for j = 1:size(xy,1)
        cx = min(max(round(xy(j,1)),1), W) + h;  cy = min(max(round(xy(j,2)),1), H) + h;
        X(:,:,:,j) = reinhard(Ip(cy-h+1:cy+h, cx-h+1:cx+h, :), ref);
    end
end

function P = reinhard(P, ref)
    lab = rgb2lab(P);
    for ch = 1:3
        v = lab(:,:,ch);
        lab(:,:,ch) = (v - mean(v(:))) / (std(v(:)) + eps) * ref.sd(ch) + ref.mu(ch);
    end
    P = im2uint8(lab2rgb(lab));
end

function [imgPath, annPath] = findFiles(s, patient, image)
    f = dir(fullfile(s{2}, patient, '**', image));
    if isempty(f), error('Photo not found: %s / %s', patient, image); end
    imgPath = fullfile(f(1).folder, f(1).name);
    [~, stem] = fileparts(image);
    annPath = fullfile(s{3}, patient, stem + ".txt");
    if ~isfile(annPath), annPath = fullfile(s{3}, patient, strrep(stem, "_", "") + ".txt"); end
    if ~isfile(annPath), annPath = ""; end
end

function obj = readAnn(f)
    Ls = strtrim(splitlines(string(fileread(f))));  Ls(Ls == "") = [];
    obj = struct('label', {}, 'x', {}, 'y', {});
    for i = 2:numel(Ls)
        t = split(Ls(i), ',');
        if numel(t) < 7, continue; end
        v = str2double(t(6:end));  xs = v(1:2:end);  ys = v(2:2:end);
        if t(4) == "Polygon", x = mean(xs); y = mean(ys); else, x = xs(1); y = ys(1); end
        obj(end+1) = struct('label', t(2), 'x', x, 'y', y); %#ok<AGROW>
    end
end

function I = toRGB8(I)
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    if size(I,3) > 3,  I = I(:,:,1:3); end
    if isa(I, 'uint16'), I = uint8(double(I) / 257);
    elseif isfloat(I) || islogical(I), I = uint8(255 * I); end
end
