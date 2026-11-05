%% CLASSIFY ONE PHOTO - demo
% Pick any thick-smear photo (jpg / tif). The trained M3 model gives the
% diagnosis: P. falciparum, P. vivax or uninfected.
% Same procedure and settings as stage6_whole_photo.m (threshold T from
% validation). Nothing is changed or saved; it only shows the result.

clear; clc;
cfg.modelFile = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full\M3_efficientnetb0_finetuned.mat';
cfg.refFile   = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing_full\stain_reference.mat';
cfg.T         = 1;       % minimum parasites to call a photo infected (chosen in stage 6)
cfg.minScore  = 0.90;
cfg.patch     = 96;
cfg.contrast  = 20;  cfg.minArea = 8;  cfg.maxArea = 6000;  cfg.maxCand = 1500;

[f, d] = uigetfile({'*.jpg;*.jpeg;*.tif;*.tiff;*.png', 'Smear photos'}, 'Pick a smear photo', ...
                   'C:\Users\kisha\PROJECTS\dsp\dataset');
if isequal(f, 0), disp('No photo picked.'); return; end
imgPath = fullfile(d, f);

L = load(cfg.modelFile);  net = L.M3;  classes = L.classes(:)';
inSz = net.Layers(1).InputSize(1:2);
R = load(cfg.refFile);  ref = R.ref;

t0 = tic;
I  = toRGB8(imread(imgPath));            % no rotation (EXIF tag ignored, as in training)
xy = findCandidates(I, cfg);
if isempty(xy)
    sc = zeros(0, numel(classes));
else
    X  = cutPatches(I, xy, cfg.patch, ref);
    sc = minibatchpredict(net, augmentedImageDatastore(inSz, X), 'MiniBatchSize', 64);
end
[mx, lab] = max(sc, [], 2);
iF = find(strcmp(classes, 'falciparum'));  iV = find(strcmp(classes, 'vivax'));
isF = lab == iF & mx >= cfg.minScore;   isV = lab == iV & mx >= cfg.minScore;
nPar = sum(isF) + sum(isV);
if nPar < cfg.T
    diagnosis = "UNINFECTED";
elseif sum(sc(isF, iF)) >= sum(sc(isV, iV))
    diagnosis = "P. FALCIPARUM";
else
    diagnosis = "P. VIVAX";
end

fprintf('\nPhoto     : %s\n', f);
fprintf('Size      : %d x %d\n', size(I,2), size(I,1));
fprintf('Candidates: %d dark spots checked\n', size(xy,1));
fprintf('Parasite-like spots: %d falciparum, %d vivax (model at least %.0f%% sure)\n', sum(isF), sum(isV), 100*cfg.minScore);
fprintf('DIAGNOSIS : %s   (%.1f s)\n', diagnosis, toc(t0));
if ~(size(I,1) == 3024 && size(I,2) == 4032)
    fprintf('Note: this photo is not 4032 x 3024 like the training photos, so the result is less reliable.\n');
end

fig = figure('Position', [50 50 1100 800], 'Color', 'w');
try, theme(fig, 'light'); catch, end
imshow(I); hold on;
plot(xy(isF,1), xy(isF,2), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
plot(xy(isV,1), xy(isV,2), 'ys', 'MarkerSize', 12, 'LineWidth', 1.5);
hold off;
title(sprintf('Diagnosis: %s   (%d falciparum-like, %d vivax-like spots)', diagnosis, sum(isF), sum(isV)), ...
      'Interpreter', 'none', 'Color', 'k');

%% ======================= helpers (same as stage 6) =======================
function xy = findCandidates(I, cfg)
    g  = im2single(rgb2gray(imresize(I, 0.5))) * 255;
    fov = imerode(imfill(g > 25, 'holes'), strel('disk', 15));
    bg = imgaussfilt(g, 25);
    bw = (bg - g) > cfg.contrast & fov;
    bw = bwareafilt(bw, [cfg.minArea cfg.maxArea]);
    st = regionprops(bw, g, 'Centroid', 'MeanIntensity');
    if isempty(st), xy = zeros(0,2); return; end
    xy = 2 * cat(1, st.Centroid);
    [~, o] = sort([st.MeanIntensity]);
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

function I = toRGB8(I)
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    if size(I,3) > 3,  I = I(:,:,1:3); end
    if isa(I, 'uint16'), I = uint8(double(I) / 257);
    elseif isfloat(I) || islogical(I), I = uint8(255 * I); end
end
