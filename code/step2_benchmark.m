%% STEP 2 - Benchmark 6 preprocessing + feature + classifier pipelines
% Uses the patches from step 1 (train / val / test split by patient).
% Classical pipelines train on train+val; CNNs use val for monitoring.
% Everything is scored on the same held-out test patients.
%
% Needs: Image Processing, Computer Vision, Statistics and Machine Learning,
%        Deep Learning and Parallel Computing toolboxes, plus the
%        ResNet-50 and EfficientNet-b0 model support packages.
%
% No rotation anywhere. Patches are already cut out, so left-right and
% up-down flips cannot misalign anything; set cfg.flips = false to skip them.

clear; clc; rng(42);

cfg.patchDir  = 'C:\Users\kisha\PROJECTS\dsp\patches_subset';
cfg.outDir    = 'C:\Users\kisha\PROJECTS\dsp\results\step2';
cfg.cnnEpochs = 12;
cfg.batch     = 32;
cfg.flips     = true;
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

%% ---------------- load patches ----------------
[Xtr, Ytr] = loadSplit(cfg.patchDir, 'train');
[Xva, Yva] = loadSplit(cfg.patchDir, 'val');
[Xte, Yte] = loadSplit(cfg.patchDir, 'test');
classes = categories(Ytr);
fprintf('Loaded %d train, %d val, %d test patches. Classes: %s\n', ...
        numel(Ytr), numel(Yva), numel(Yte), strjoin(classes, ', '));

% colour reference for stain normalisation = mean Lab stats of training patches
ref = labStats(Xtr);

%% ---------------- preprocessing variants ----------------
pre.raw      = @(X) X;
pre.clahe    = @(X) mapPatches(X, @claheL);
pre.reinhard = @(X) mapPatches(X, @(P) reinhard(P, ref));

%% ---------------- pipelines ----------------
%  name                         preprocessing  method
P = {
 'P1 Colour+GLCM -> SVM',        'raw',      'colourglcm_svm'
 'P2 HOG+LBP -> SVM',            'clahe',    'hoglbp_svm'
 'P3 Colour+GLCM+HOG+LBP -> RF', 'reinhard', 'all_rf'
 'P4 ResNet-50 features -> SVM', 'reinhard', 'resnet_feat_svm'
 'P5 ResNet-50 fine-tuned',      'reinhard', 'finetune_resnet50'
 'P6 EfficientNet-b0 fine-tuned','reinhard', 'finetune_efficientnetb0'
};

R = table();
for i = 1:size(P,1)
    name = P{i,1}; pp = P{i,2}; method = P{i,3};
    fprintf('\n===== %s  (preprocessing: %s) =====\n', name, pp);
    f = pre.(pp);
    A = f(Xtr); B = f(Xva); C = f(Xte);
    t0 = tic;
    switch method
        case 'colourglcm_svm'
            AB  = cat(4, A, B);
            Ytp = trainSVM([featColour(AB) featGLCM(AB)], [Ytr; Yva], [featColour(C) featGLCM(C)]);
        case 'hoglbp_svm'
            AB  = cat(4, A, B);
            Ftr = [featHOG(AB) featLBP(AB)];
            Fte = [featHOG(C) featLBP(C)];
            Ytp = trainSVM(Ftr, [Ytr; Yva], Fte);
        case 'all_rf'
            AB  = cat(4, A, B);
            Ftr = [featColour(AB) featGLCM(AB) featHOG(AB) featLBP(AB)];
            Fte = [featColour(C)  featGLCM(C)  featHOG(C)  featLBP(C)];
            rf  = TreeBagger(300, Ftr, [Ytr; Yva], 'Method', 'classification');
            Ytp = categorical(predict(rf, Fte), classes);
        case 'resnet_feat_svm'
            net = imagePretrainedNetwork("resnet50");
            Ftr = deepFeatures(net, cat(4, A, B), cfg.batch);
            Fte = deepFeatures(net, C, cfg.batch);
            Ytp = trainSVM(Ftr, [Ytr; Yva], Fte);
        case {'finetune_resnet50', 'finetune_efficientnetb0'}
            arch = erase(method, 'finetune_');
            Ytp = fineTune(arch, A, Ytr, B, Yva, C, classes, cfg, ...
                           fullfile(cfg.outDir, [arch '_trained.mat']));
    end
    secs = toc(t0);
    m = scoreIt(Yte, Ytp, classes);
    fprintf('Accuracy %.2f%%  macro-F1 %.3f  (%.0f s)\n', 100*m.acc, m.macroF1, secs);

    fig = figure('Visible', 'off');
    confusionchart(Yte, Ytp, 'RowSummary', 'row-normalized', 'Title', name);
    exportgraphics(fig, fullfile(cfg.outDir, sprintf('confusion_P%d.png', i)));
    close(fig);

    row = table(string(name), string(pp), 100*m.acc, m.macroF1, ...
                100*m.recall(1), 100*m.recall(2), 100*m.recall(3), secs, ...
        'VariableNames', {'Pipeline','Preprocessing','Accuracy','MacroF1', ...
        ['Recall_' classes{1}], ['Recall_' classes{2}], ['Recall_' classes{3}], 'Seconds'});
    R = [R; row]; %#ok<AGROW>
    writetable(R, fullfile(cfg.outDir, 'benchmark_results.csv'));   % saved after each pipeline
end

R = sortrows(R, 'MacroF1', 'descend');
disp(R);
writetable(R, fullfile(cfg.outDir, 'benchmark_results.csv'));
fprintf('Results and confusion matrices saved in %s\n', cfg.outDir);

%% ======================= helper functions =======================
function [X, Y] = loadSplit(root, split)
    X = []; Y = strings(0,1);
    cls = dir(fullfile(root, split));
    cls = cls([cls.isdir] & ~startsWith({cls.name}, '.'));
    for c = 1:numel(cls)
        f = dir(fullfile(root, split, cls(c).name, '*.png'));
        Xi = zeros(96, 96, 3, numel(f), 'uint8');
        for k = 1:numel(f)
            Xi(:,:,:,k) = imread(fullfile(f(k).folder, f(k).name));
        end
        X = cat(4, X, Xi);
        Y = [Y; repmat(string(cls(c).name), numel(f), 1)]; %#ok<AGROW>
    end
    Y = categorical(Y);
end

function Y = mapPatches(X, fn)
    Y = X;
    for k = 1:size(X,4), Y(:,:,:,k) = fn(X(:,:,:,k)); end
end

function P = claheL(P)
    lab = rgb2lab(P);
    lab(:,:,1) = 100 * adapthisteq(lab(:,:,1)/100, 'ClipLimit', 0.01, 'NumTiles', [4 4]);
    P = im2uint8(lab2rgb(lab));
end

function s = labStats(X)
    lab = rgb2lab(reshape(permute(X, [1 2 4 3]), [], 1, 3));
    lab = reshape(lab, [], 3);
    s.mu = mean(lab); s.sd = std(lab);
end

function P = reinhard(P, ref)
% Reinhard colour normalisation: match each patch's Lab mean/std to the reference
    lab = rgb2lab(P);
    for ch = 1:3
        v = lab(:,:,ch);
        lab(:,:,ch) = (v - mean(v(:))) / (std(v(:)) + eps) * ref.sd(ch) + ref.mu(ch);
    end
    P = im2uint8(lab2rgb(lab));
end

function F = featColour(X)
    n = size(X,4); F = zeros(n, 54);
    for k = 1:n
        P = X(:,:,:,k);
        hsv = rgb2hsv(P); lab = rgb2lab(P);
        h = [];
        for ch = 1:3, h = [h histcounts(hsv(:,:,ch), 16, 'BinLimits', [0 1], 'Normalization', 'probability')]; end %#ok<AGROW>
        l = reshape(lab, [], 3);
        F(k,:) = [h mean(l) std(l)];
    end
end

function F = featGLCM(X)
    n = size(X,4); F = zeros(n, 16);
    off = [0 1; -1 1; -1 0; -1 -1];
    for k = 1:n
        g = graycomatrix(rgb2gray(X(:,:,:,k)), 'Offset', off, 'NumLevels', 32, 'Symmetric', true);
        s = graycoprops(g);
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

function Yp = trainSVM(Ftr, Ytr, Fte)
    t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
    mdl = fitcecoc(Ftr, Ytr, 'Learners', t);
    Yp = predict(mdl, Fte);
end

function F = deepFeatures(net, X, bs)
    sz = net.Layers(1).InputSize(1:2);
    ds = augmentedImageDatastore(sz, X);
    F  = minibatchpredict(net, ds, 'Outputs', 'avg_pool', 'MiniBatchSize', bs);
    F  = double(squeeze(F));
    if size(F,1) ~= size(X,4), F = F.'; end
end

function Yp = fineTune(arch, Xtr, Ytr, Xva, Yva, Xte, classes, cfg, saveFile)
    net = imagePretrainedNetwork(arch, 'NumClasses', numel(classes));
    sz  = net.Layers(1).InputSize(1:2);
    aug = imageDataAugmenter('RandXReflection', cfg.flips, 'RandYReflection', cfg.flips);
    dsTr = augmentedImageDatastore(sz, Xtr, Ytr, 'DataAugmentation', aug);
    dsVa = augmentedImageDatastore(sz, Xva, Yva);
    opts = trainingOptions('adam', ...
        'InitialLearnRate', 1e-4, 'MaxEpochs', cfg.cnnEpochs, ...
        'MiniBatchSize', cfg.batch, 'Shuffle', 'every-epoch', ...
        'ValidationData', dsVa, 'ValidationFrequency', 20, ...
        'Metrics', 'accuracy', 'OutputNetwork', 'best-validation', ...
        'ExecutionEnvironment', 'auto', 'Plots', 'training-progress', 'Verbose', true);
    net = trainnet(dsTr, net, 'crossentropy', opts);
    save(saveFile, 'net', 'classes');
    scores = minibatchpredict(net, augmentedImageDatastore(sz, Xte), 'MiniBatchSize', cfg.batch);
    Yp = scores2label(scores, classes);
end

function m = scoreIt(Yt, Yp, classes)
    Yt = categorical(Yt, classes); Yp = categorical(Yp, classes);
    C  = confusionmat(Yt, Yp, 'Order', classes);
    tp = diag(C)';
    prec = tp ./ max(sum(C,1), 1);
    rec  = tp ./ max(sum(C,2)', 1);
    f1   = 2*prec.*rec ./ max(prec+rec, eps);
    m.acc = sum(tp) / sum(C(:));
    m.recall = rec; m.macroF1 = mean(f1);
end
