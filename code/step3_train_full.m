%% STEP 3 - Train the winning pipeline (Reinhard + EfficientNet-b0) on the full data
% Before running: re-run step 1 with these settings to build the big patch set
%     cfg.outDir         = 'C:\Users\kisha\PROJECTS\dsp\patches_full';
%     cfg.targetPerClass = 25000;   % about 25k patches per class (balanced)
%     cfg.maxPerImage    = 30;
% That gives about 75k patches (roughly 1.5 GB) and takes 20-40 minutes.
%
% This script streams patches from disk, so it does not need them all in RAM.
% No rotation: only left-right / up-down flips of already-cut patches.

clear; clc; rng(42);

cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\patches_full';
cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\results\step3';
cfg.modelOut = 'C:\Users\kisha\PROJECTS\dsp\models\effnetb0_full.mat';
cfg.epochs   = 8;
cfg.batch    = 32;      % lower to 16 if you get a GPU out-of-memory error
for d = {cfg.outDir, fileparts(cfg.modelOut)}
    if ~isfolder(d{1}), mkdir(d{1}); end
end

%% ---------------- datastores ----------------
mk = @(s) imageDatastore(fullfile(cfg.patchDir, s), ...
        'IncludeSubfolders', true, 'LabelSource', 'foldernames');
dsTr = mk('train'); dsVa = mk('val'); dsTe = mk('test');
classes = categories(dsTr.Labels);
disp(countEachLabel(dsTr));

% stain reference from 2000 random training patches
pick = randperm(numel(dsTr.Files), min(2000, numel(dsTr.Files)));
S = zeros(96, 96, 3, numel(pick), 'uint8');
for k = 1:numel(pick), S(:,:,:,k) = imread(dsTr.Files{pick(k)}); end
lab = reshape(rgb2lab(reshape(permute(S, [1 2 4 3]), [], 1, 3)), [], 3);
ref.mu = mean(lab); ref.sd = std(lab);

% apply Reinhard normalisation every time a patch is read
for ds = {dsTr, dsVa, dsTe}
    ds{1}.ReadFcn = @(f) reinhard(imread(f), ref);
end

%% ---------------- network + training ----------------
net = imagePretrainedNetwork("efficientnetb0", 'NumClasses', numel(classes));
sz  = net.Layers(1).InputSize(1:2);
aug = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true);
augTr = augmentedImageDatastore(sz, dsTr, 'DataAugmentation', aug);
augVa = augmentedImageDatastore(sz, dsVa);
augTe = augmentedImageDatastore(sz, dsTe);

itersPerEpoch = floor(numel(dsTr.Files) / cfg.batch);
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 3, 'LearnRateDropFactor', 0.3, ...
    'MaxEpochs', cfg.epochs, 'MiniBatchSize', cfg.batch, 'Shuffle', 'every-epoch', ...
    'ValidationData', augVa, 'ValidationFrequency', max(1, floor(itersPerEpoch/2)), ...
    'ValidationPatience', 4, 'OutputNetwork', 'best-validation', ...
    'Metrics', 'accuracy', 'Plots', 'training-progress', 'Verbose', true, ...
    'ExecutionEnvironment', 'auto');

tic;
net = trainnet(augTr, net, 'crossentropy', opts);
trainMinutes = toc/60;
save(cfg.modelOut, 'net', 'classes', 'ref');
fprintf('Training took %.1f min. Model saved to %s\n', trainMinutes, cfg.modelOut);

%% ---------------- patch-level test results ----------------
scores = minibatchpredict(net, augTe, 'MiniBatchSize', cfg.batch);
Yp = scores2label(scores, classes);
Yt = dsTe.Labels;
m  = scoreIt(Yt, Yp, classes);
fprintf('\nTEST (patch level): accuracy %.2f%%, macro-F1 %.3f\n', 100*m.acc, m.macroF1);
for c = 1:numel(classes)
    fprintf('  %-11s precision %.1f%%  recall %.1f%%  F1 %.3f\n', classes{c}, ...
            100*m.prec(c), 100*m.recall(c), m.f1(c));
end
fig = figure;
confusionchart(Yt, Yp, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized', ...
               'Title', 'EfficientNet-b0, full data, test patients');
exportgraphics(fig, fullfile(cfg.outDir, 'confusion_test.png'));

%% ---------------- patient-level test results ----------------
% Average the class scores of every test patch from one patient.
% NOTE: optimistic, because these patches were cut at annotated spots.
% The real slide-level pipeline (find candidates first) is step 4.
[~, names] = cellfun(@fileparts, dsTe.Files, 'UniformOutput', false);
patient = extractBefore(string(names), "__");
[G, pid] = findgroups(patient);
pScore  = splitapply(@(s) mean(s, 1), scores, G);
pPred   = scores2label(pScore, classes);
pTrue   = splitapply(@(y) y(1), Yt, G);
fprintf('TEST (patient level): %d of %d patients correct (%.1f%%)\n', ...
        sum(pPred == pTrue), numel(pid), 100*mean(pPred == pTrue));
writetable(table(pid, pTrue, pPred, 'VariableNames', {'patient','true','predicted'}), ...
           fullfile(cfg.outDir, 'patient_level_test.csv'));

%% ---------------- Grad-CAM: where is the model looking? ----------------
fig = figure('Position', [100 100 1200 420]);
tiledlayout(2, 6, 'TileSpacing', 'compact');
for c = 1:numel(classes)
    idx = find(Yt == classes{c});
    idx = idx(randperm(numel(idx), min(2, numel(idx))));
    for k = idx'
        I = imresize(readimage(dsTe, k), sz);
        map = gradCAM(net, single(I), c);
        nexttile; imshow(I); title(sprintf('%s', classes{c}), 'FontSize', 8);
        nexttile; imshow(I); hold on;
        imagesc(map, 'AlphaData', 0.5); colormap jet; hold off;
        title(sprintf('pred: %s', string(Yp(k))), 'FontSize', 8);
    end
end
exportgraphics(fig, fullfile(cfg.outDir, 'gradcam_examples.png'));
fprintf('Figures and tables saved in %s\n', cfg.outDir);

%% ======================= helpers =======================
function P = reinhard(P, ref)
    lab = rgb2lab(P);
    for ch = 1:3
        v = lab(:,:,ch);
        lab(:,:,ch) = (v - mean(v(:))) / (std(v(:)) + eps) * ref.sd(ch) + ref.mu(ch);
    end
    P = im2uint8(lab2rgb(lab));
end

function m = scoreIt(Yt, Yp, classes)
    C  = confusionmat(categorical(Yt, classes), categorical(Yp, classes), 'Order', classes);
    tp = diag(C)';
    m.prec   = tp ./ max(sum(C,1), 1);
    m.recall = tp ./ max(sum(C,2)', 1);
    m.f1     = 2*m.prec.*m.recall ./ max(m.prec + m.recall, eps);
    m.acc    = sum(tp) / sum(C(:));
    m.macroF1 = mean(m.f1);
end
