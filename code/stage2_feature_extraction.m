%% STAGE 2 - FEATURE EXTRACTION
% Input : Reinhard-normalised patches from stage 1 (train / val / test by patient)
% Output: feature matrices for every patch, saved for the training stage,
%         plus figures and a summary table.
%
% Two kinds of features are extracted:
%   A) Handcrafted features (1,060 values per patch)
%      - Colour : HSV histograms (3 x 16 bins) + Lab mean and std      = 54
%      - Texture: GLCM contrast/correlation/energy/homogeneity, 4 dirs = 16
%      - Shape  : HOG (histogram of oriented gradients), 16x16 cells   = 900
%      - Texture: LBP (local binary patterns), 32x32 cells             = 90
%   B) Deep features (1,280 values per patch)
%      - EfficientNet-b0 pretrained on ImageNet, output of the final
%        global average pooling layer (the selected model's backbone)
%
% How well each feature set separates the 3 classes is measured with the
% silhouette score (range -1 to 1, higher = classes better separated)
% and shown with t-SNE plots. No classifier is trained in this stage.

clear; clc; rng(42);

cfg.fullData = true;   % true = full dataset run, false = 1,300-patch sample
if cfg.fullData
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing_full\patches_reinhard';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage2_features_full';
else
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing\patches_reinhard';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage2_features';
end
cfg.maxPlot  = 3000;   % silhouette and t-SNE use at most this many random training patches
cfg.batch    = 32;
cfg.chunk    = 2000;   % patches processed (and progress printed) per chunk
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

splits = ["train" "val" "test"];
D = struct();
for s = splits
    D.(s) = imageDatastore(fullfile(cfg.patchDir, s), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end
classes = categories(D.train.Labels);

%% ---------------- CNN backbone ----------------
net = imagePretrainedNetwork("efficientnetb0");
inSz = net.Layers(1).InputSize(1:2);
isGAP  = arrayfun(@(L) isa(L, 'nnet.cnn.layer.GlobalAveragePooling2DLayer'), net.Layers);
gapName = net.Layers(find(isGAP, 1, 'last')).Name;
isConv = arrayfun(@(L) isa(L, 'nnet.cnn.layer.Convolution2DLayer'), net.Layers);
conv1Name = net.Layers(find(isConv, 1, 'first')).Name;
fprintf('Deep features taken from layer: %s\n', gapName);

%% ---------------- extract features for every split ----------------
summary = table();
tHand = 0; tDeep = 0;
nAll = sum(arrayfun(@(s) numel(D.(s).Files), splits));  nDone = 0;  tStart = tic;
for s = splits
    ds = D.(s);
    Y = ds.Labels;
    [~, names] = cellfun(@fileparts, ds.Files, 'UniformOutput', false);
    patient = extractBefore(string(names), "__");
    n = numel(ds.Files);
    Fcolour = []; Fglcm = []; Fhog = []; Flbp = []; Fdeep = [];

    for c0 = 1:cfg.chunk:n          % work in chunks so progress can be printed
        idx = c0:min(c0+cfg.chunk-1, n);
        X = readall(subset(ds, idx));  X = cat(4, X{:});

        t0 = tic;
        Fcolour = [Fcolour; featColour(X)]; %#ok<AGROW>
        Fglcm   = [Fglcm;   featGLCM(X)];   %#ok<AGROW>
        Fhog    = [Fhog;    featHOG(X)];    %#ok<AGROW>
        Flbp    = [Flbp;    featLBP(X)];    %#ok<AGROW>
        tHand = tHand + toc(t0);

        t0 = tic;
        Fd = minibatchpredict(net, augmentedImageDatastore(inSz, X), ...
                              'Outputs', gapName, 'MiniBatchSize', cfg.batch);
        Fd = double(squeeze(Fd));
        if size(Fd, 1) ~= size(X, 4), Fd = Fd.'; end
        Fdeep = [Fdeep; Fd]; %#ok<AGROW>
        tDeep = tDeep + toc(t0);

        nDone = nDone + numel(idx);
        el = toc(tStart);
        fprintf('  %-5s %6d / %6d   |  overall %6d / %6d (%4.1f%%)  elapsed %5.1f min, about %5.1f min left\n', ...
                s, idx(end), n, nDone, nAll, 100*nDone/nAll, el/60, el/nDone*(nAll-nDone)/60);
    end

    Fhand = [Fcolour Fglcm Fhog Flbp];
    files = ds.Files;
    save(fullfile(cfg.outDir, "features_" + s + ".mat"), ...
         'Fcolour', 'Fglcm', 'Fhog', 'Flbp', 'Fhand', 'Fdeep', 'Y', 'patient', 'files', 'classes');
    fprintf('%-5s: %d patches -> handcrafted %d features, deep %d features\n', ...
            s, n, size(Fhand,2), size(Fdeep,2));
    if s == "train"
        tr = struct('Fcolour', Fcolour, 'Fglcm', Fglcm, 'Fhog', Fhog, 'Flbp', Flbp, ...
                    'Fhand', Fhand, 'Fdeep', Fdeep, 'Y', Y);
    end
end

%% ---------------- feature summary (on training patients) ----------------
fprintf('\nComputing silhouette scores (takes a few minutes)...\n');
groups = {'Colour (HSV hist + Lab stats)', tr.Fcolour;
          'GLCM texture',                  tr.Fglcm;
          'HOG',                           tr.Fhog;
          'LBP',                           tr.Flbp;
          'All handcrafted combined',      tr.Fhand;
          'EfficientNet-b0 deep features', tr.Fdeep};
nTrain = numel(tr.Y);
idxS = randperm(nTrain, min(cfg.maxPlot, nTrain));   % subsample for silhouette / t-SNE
for g = 1:size(groups, 1)
    F = groups{g, 2};
    summary = [summary; table(string(groups{g,1}), size(F,2), ...
        round(separability(F(idxS,:), tr.Y(idxS)), 3), ...
        'VariableNames', {'FeatureSet', 'Dimensions', 'Silhouette'})]; %#ok<AGROW>
end
fprintf('\nExtraction time: handcrafted %.1f s, deep %.1f s (all splits)\n', tHand, tDeep);
fprintf('\n===== FEATURE SUMMARY (training patients, %d patches; silhouette on %d random) =====\n', nTrain, numel(idxS));
disp(summary);
writetable(summary, fullfile(cfg.outDir, 'feature_summary.csv'));

%% ---------------- figure 1: handcrafted features on one patch per class ----------------
fig = figure('Position', [50 50 1100 750], 'Color', 'w');
tiledlayout(numel(classes), 4, 'TileSpacing', 'compact');
for c = 1:numel(classes)
    k = find(tr.Y == classes{c}, 1);
    P = readimage(D.train, k);  g = rgb2gray(P);
    nexttile; imshow(P); title(classes{c});
    nexttile; [~, vis] = extractHOGFeatures(g, 'CellSize', [16 16]);
    imshow(g); hold on; plot(vis, 'Color', 'y'); hold off; title('HOG');
    nexttile; bar(extractLBPFeatures(g, 'Upright', false), 'k'); title('LBP histogram'); xlim([0 11]);
    nexttile; hsv = rgb2hsv(P);
    histogram(hsv(:,:,1), 16, 'BinLimits', [0 1], 'FaceColor', [0.3 0.3 0.3]); title('Hue histogram');
end
exportgraphics(fig, fullfile(cfg.outDir, 'fig_handcrafted_features.png'));

%% ---------------- figure 2: what the CNN sees (first conv layer) ----------------
fig = figure('Position', [50 50 1100 450], 'Color', 'w');
tiledlayout(numel(classes), 9, 'TileSpacing', 'tight');
for c = 1:numel(classes)
    k = find(tr.Y == classes{c}, 1);
    P = imresize(readimage(D.train, k), inSz);
    A = minibatchpredict(net, single(P), 'Outputs', conv1Name);
    nexttile; imshow(P); title(classes{c});
    for ch = 1:8
        nexttile; imshow(rescale(A(:,:,ch)));
    end
end
exportgraphics(fig, fullfile(cfg.outDir, 'fig_cnn_feature_maps.png'));

%% ---------------- figure 3: t-SNE of handcrafted vs deep features ----------------
fprintf('Running t-SNE (takes a few minutes)...\n');
fig = figure('Position', [50 50 1100 450], 'Color', 'w');
tiledlayout(1, 2);
for g = [5 6]
    Z = tsne(zscore(groups{g,2}(idxS,:)), 'NumPCAComponents', 50, 'Perplexity', 30);
    nexttile; gscatter(Z(:,1), Z(:,2), tr.Y(idxS), [], 'o+x', 5);
    title(sprintf('%s (silhouette %.3f)', groups{g,1}, summary.Silhouette(g)));
    axis square; box on;
end
exportgraphics(fig, fullfile(cfg.outDir, 'fig_tsne.png'));
fprintf('Stage 2 done. Features and figures saved in %s\n', cfg.outDir);

%% ======================= helpers =======================
function s = separability(F, Y)
% mean silhouette after standardising and reducing to at most 50 PCA dims
    F = zscore(F);  F(:, any(isnan(F), 1)) = [];
    [~, sc] = pca(F, 'NumComponents', min(50, size(F,2)));
    s = mean(silhouette(sc, Y));
end

function F = featColour(X)
    n = size(X,4); F = zeros(n, 54);
    for k = 1:n
        P = X(:,:,:,k);  hsv = rgb2hsv(P);  lab = reshape(rgb2lab(P), [], 3);
        h = [];
        for ch = 1:3
            h = [h histcounts(hsv(:,:,ch), 16, 'BinLimits', [0 1], 'Normalization', 'probability')]; %#ok<AGROW>
        end
        F(k,:) = [h mean(lab) std(lab)];
    end
end

function F = featGLCM(X)
    n = size(X,4); F = zeros(n, 16);
    off = [0 1; -1 1; -1 0; -1 -1];
    for k = 1:n
        G = graycomatrix(rgb2gray(X(:,:,:,k)), 'Offset', off, 'NumLevels', 32, 'Symmetric', true);
        s = graycoprops(G);
        F(k,:) = [s.Contrast s.Correlation s.Energy s.Homogeneity];
    end
end

function F = featHOG(X)
    n = size(X,4);
    f1 = extractHOGFeatures(rgb2gray(X(:,:,:,1)), 'CellSize', [16 16]);
    F = zeros(n, numel(f1)); F(1,:) = f1;
    for k = 2:n, F(k,:) = extractHOGFeatures(rgb2gray(X(:,:,:,k)), 'CellSize', [16 16]); end
end

function F = featLBP(X)
    n = size(X,4);
    f1 = extractLBPFeatures(rgb2gray(X(:,:,:,1)), 'CellSize', [32 32], 'Upright', false);
    F = zeros(n, numel(f1)); F(1,:) = f1;
    for k = 2:n, F(k,:) = extractLBPFeatures(rgb2gray(X(:,:,:,k)), 'CellSize', [32 32], 'Upright', false); end
end
