%% STAGE 3 - TRAINING
% Trains three models on the TRAINING patients only:
%   M1  SVM (RBF kernel)        on handcrafted features from stage 2  (baseline)
%   M2  SVM (RBF kernel)        on EfficientNet-b0 deep features from stage 2
%   M3  EfficientNet-b0 fine-tuned end to end on Reinhard patches   (main model)
% The validation set is only used to monitor M3 during training and keep its
% best epoch. Validation and test results are reported in stages 4 and 5.
%
% No rotation: augmentation for M3 is left-right and up-down flips only.

clear; clc; rng(42);

cfg.fullData = true;   % true = full dataset run, false = 1,300-patch sample
if cfg.fullData
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features_full';
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing_full\patches_reinhard';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full';
    cfg.epochs   = 35;      % best epoch (lowest validation loss) is kept
    cfg.lrDropEvery = 10;   % learning rate x0.3 at epochs 10 and 20
    cfg.svmMaxRows = 15000; % SVM baselines use a random 15k training patches (RBF SVM on 45k is very slow)
else
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features';
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing\patches_reinhard';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage3_models';
    cfg.epochs   = 40;      % best epoch is kept, so extra epochs cannot make the saved model worse
    cfg.lrDropEvery = 15;   % learning rate x0.3 every 15 epochs
    cfg.svmMaxRows = Inf;
end
cfg.batch    = 32;
cfg.lr       = 1e-4;
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

tr = load(fullfile(cfg.featDir, 'features_train.mat'));
classes = tr.classes;
cnt = string(classes(:)) + " = " + string(countcats(tr.Y(:)));
fprintf('Training patches: %d  (%s)\n', numel(tr.Y), strjoin(cnt.', ', '));
results = table();

rs = randperm(numel(tr.Y), min(cfg.svmMaxRows, numel(tr.Y)));   % rows used by the SVM baselines
%% ---------------- M1: SVM on handcrafted features ----------------
fprintf('[1/3] Training M1: SVM on handcrafted features (%d patches, a few minutes, no progress shown)...\n', numel(rs));
t0 = tic;
svmT = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
M1 = fitcecoc(tr.Fhand(rs,:), tr.Y(rs), 'Learners', svmT, 'Coding', 'onevsone');
t1 = toc(t0);
acc1 = mean(predict(M1, tr.Fhand(rs,:)) == tr.Y(rs));
save(fullfile(cfg.outDir, 'M1_svm_handcrafted.mat'), 'M1', 'classes');
results = [results; table("M1 SVM (RBF), handcrafted features", size(tr.Fhand,2), ...
    "3 one-vs-one SVMs", 100*acc1, t1, 'VariableNames', ...
    {'Model','InputFeatures','Trainable','TrainAccuracy','TrainSeconds'})];

%% ---------------- M2: SVM on deep features ----------------
fprintf('      done in %.1f min\n[2/3] Training M2: SVM on deep features (%d patches, a few minutes)...\n', t1/60, numel(rs));
t0 = tic;
M2 = fitcecoc(tr.Fdeep(rs,:), tr.Y(rs), 'Learners', svmT, 'Coding', 'onevsone');
t2 = toc(t0);
acc2 = mean(predict(M2, tr.Fdeep(rs,:)) == tr.Y(rs));
save(fullfile(cfg.outDir, 'M2_svm_deep.mat'), 'M2', 'classes');
results = [results; table("M2 SVM (RBF), EfficientNet-b0 features", size(tr.Fdeep,2), ...
    "3 one-vs-one SVMs", 100*acc2, t2, 'VariableNames', results.Properties.VariableNames)];

%% ---------------- M3: fine-tuned EfficientNet-b0 ----------------
fprintf('      done in %.1f min\n[3/3] Fine-tuning M3: EfficientNet-b0, %d epochs (progress window + iteration table below)...\n', t2/60, cfg.epochs);
mk = @(s) imageDatastore(fullfile(cfg.patchDir, s), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
dsTr = mk('train');  dsVa = mk('val');
net = imagePretrainedNetwork("efficientnetb0", 'NumClasses', numel(classes));
inSz = net.Layers(1).InputSize(1:2);
nParams = sum(cellfun(@numel, net.Learnables.Value));

aug  = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true);
augTr = augmentedImageDatastore(inSz, dsTr, 'DataAugmentation', aug);
augVa = augmentedImageDatastore(inSz, dsVa);
opts = trainingOptions('adam', ...
    'InitialLearnRate', cfg.lr, 'MaxEpochs', cfg.epochs, 'MiniBatchSize', cfg.batch, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', cfg.lrDropEvery, 'LearnRateDropFactor', 0.3, ...
    'Shuffle', 'every-epoch', 'ValidationData', augVa, 'ValidationFrequency', max(20, floor(numel(dsTr.Files)/cfg.batch)), ...
    'OutputNetwork', 'best-validation', 'Metrics', 'accuracy', ...
    'Plots', 'training-progress', 'Verbose', true, 'ExecutionEnvironment', 'auto');

t0 = tic;
[M3, info] = trainnet(augTr, net, 'crossentropy', opts);
t3 = toc(t0);
scores = minibatchpredict(M3, augmentedImageDatastore(inSz, dsTr), 'MiniBatchSize', cfg.batch);
acc3 = mean(scores2label(scores, classes) == dsTr.Labels);
save(fullfile(cfg.outDir, 'M3_efficientnetb0_finetuned.mat'), 'M3', 'classes', 'info', 'cfg');
results = [results; table("M3 EfficientNet-b0 fine-tuned", inSz(1)*inSz(2)*3, ...
    sprintf("%.2f million weights", nParams/1e6), 100*acc3, t3, 'VariableNames', results.Properties.VariableNames)];

%% ---------------- training curves of M3 (saved as a figure) ----------------
TH = info.TrainingHistory;  VH = info.ValidationHistory;
fig = figure('Position', [50 50 1000 650], 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile; plot(TH.Iteration, TH.Accuracy, 'Color', [0.6 0.6 0.6]); hold on;
plot(VH.Iteration, VH.Accuracy, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold off;
ylabel('Accuracy (%)'); legend('Training', 'Validation', 'Location', 'southeast'); grid on;
title('EfficientNet-b0 fine-tuning');
nexttile; plot(TH.Iteration, TH.Loss, 'Color', [0.6 0.6 0.6]); hold on;
plot(VH.Iteration, VH.Loss, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold off;
ylabel('Cross-entropy loss'); xlabel('Iteration'); legend('Training', 'Validation'); grid on;
exportgraphics(fig, fullfile(cfg.outDir, 'fig_training_curves.png'));

%% ---------------- summary ----------------
results.TrainAccuracy = round(results.TrainAccuracy, 2);
results.TrainSeconds  = round(results.TrainSeconds, 1);
fprintf('\n===== TRAINING SUMMARY (training patients only) =====\n');
disp(results);
writetable(results, fullfile(cfg.outDir, 'training_summary.csv'));

if canUseGPU, g = gpuDevice; hw = "GPU: " + string(g.Name); else, hw = "CPU"; end
hp = table(["Optimiser"; "Initial learning rate (x0.3 every " + cfg.lrDropEvery + " epochs)"; "Epochs"; "Mini-batch size"; "Input size"; ...
            "Loss"; "Augmentation"; "Best epoch kept by"; "Hardware"], ...
           ["Adam"; string(cfg.lr); string(cfg.epochs); string(cfg.batch); sprintf("%d x %d x 3", inSz); ...
            "Cross-entropy"; "Random left-right and up-down flips (no rotation)"; "Lowest validation loss"; ...
            hw], 'VariableNames', {'Setting', 'Value'});
fprintf('\n===== M3 HYPERPARAMETERS =====\n');
disp(hp);
writetable(hp, fullfile(cfg.outDir, 'hyperparameters.csv'));
fprintf('Stage 3 done. Models and figures saved in %s\n', cfg.outDir);
