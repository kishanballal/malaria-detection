%% STAGE 4 - VALIDATION
% Evaluates the three models from stage 3 on the VALIDATION patients
% (54 patients the models were never trained on).
%   M1  SVM on handcrafted features
%   M2  SVM on EfficientNet-b0 deep features
%   M3  EfficientNet-b0 fine-tuned (main model)
% Reports patch-level accuracy, precision, recall, F1 and AUC per class,
% patient-level accuracy (majority vote of a patient's patches),
% confusion matrices, ROC curves and wrongly classified examples.
% The test set is NOT touched here (that is stage 5).

clear; clc; rng(42);

cfg.fullData = true;   % true = full dataset run, false = 1,300-patch sample
if cfg.fullData
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features_full';
    cfg.modelDir = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage4_validation_full';
else
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features';
    cfg.modelDir = 'C:\Users\kisha\PROJECTS\dsp\stage3_models';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage4_validation';
end
cfg.batch = 32;
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

%% ---------------- load models and validation data ----------------
fprintf('Loading models and validation features...\n');
va = load(fullfile(cfg.featDir, 'features_val.mat'));
L1 = load(fullfile(cfg.modelDir, 'M1_svm_handcrafted.mat'));
L2 = load(fullfile(cfg.modelDir, 'M2_svm_deep.mat'));
L3 = load(fullfile(cfg.modelDir, 'M3_efficientnetb0_finetuned.mat'));
classes = L3.classes(:)';
yt = va.Y(:);
fprintf('Validation: %d patches from %d patients\n', numel(yt), numel(unique(va.patient)));

%% ---------------- predictions ----------------
fprintf('[1/3] M1 predicting...\n');
[yp1, s1] = predict(L1.M1, va.Fhand);
fprintf('[2/3] M2 predicting...\n');
[yp2, s2] = predict(L2.M2, va.Fdeep);
fprintf('[3/3] M3 predicting on %d patches (GPU)...\n', numel(yt));
inSz = L3.M3.Layers(1).InputSize(1:2);
ds = imageDatastore(va.files);                     % same order as the feature file
s3 = minibatchpredict(L3.M3, augmentedImageDatastore(inSz, ds), 'MiniBatchSize', cfg.batch);
yp3 = scores2label(s3, classes);

names  = ["M1 SVM, handcrafted features"; "M2 SVM, EfficientNet-b0 features"; "M3 EfficientNet-b0 fine-tuned"];
preds  = {yp1(:), yp2(:), yp3(:)};
scores = {s1, s2, s3};

%% ---------------- metrics ----------------
summary = table();  perClass = table();  CM = cell(1, 3);
for m = 1:3
    yp = categorical(cellstr(string(preds{m})), classes);
    ytc = categorical(cellstr(string(yt)), classes);
    C = confusionmat(ytc, yp, 'Order', categorical(classes));
    prec = diag(C) ./ max(sum(C,1)', 1);
    rec  = diag(C) ./ max(sum(C,2), 1);
    f1   = 2*prec.*rec ./ max(prec+rec, eps);
    auc  = zeros(numel(classes), 1);
    for c = 1:numel(classes)
        [~, ~, ~, auc(c)] = perfcurve(ytc == classes{c}, scores{m}(:,c), true);
    end
    % patient level: a patient is labelled with the majority vote of its patches
    [pats, ~, g] = unique(va.patient(:));
    pt = splitapply(@(x) mode(x), ytc, g);
    pp = splitapply(@(x) mode(x), yp,  g);
    summary = [summary; table(names(m), round(100*mean(yp == ytc), 2), round(100*mean(f1), 2), ...
        round(mean(auc), 4), sprintf("%d / %d", sum(pp == pt), numel(pats)), round(100*mean(pp == pt), 1), ...
        'VariableNames', {'Model','PatchAccuracy','MacroF1','MeanAUC','PatientsCorrect','PatientAccuracy'})]; %#ok<AGROW>
    perClass = [perClass; table(repmat(names(m), numel(classes), 1), string(classes(:)), sum(C,2), ...
        round(100*prec, 2), round(100*rec, 2), round(100*f1, 2), round(auc, 4), ...
        'VariableNames', {'Model','Class','Patches','Precision','Recall','F1','AUC'})]; %#ok<AGROW>
    CM{m} = C;
    preds{m} = yp;
end

fprintf('\n===== VALIDATION RESULTS (%d patches, %d patients) =====\n', numel(yt), numel(unique(va.patient)));
disp(summary);
fprintf('\n===== PER CLASS =====\n');
disp(perClass);
writetable(summary,  fullfile(cfg.outDir, 'validation_summary.csv'));
writetable(perClass, fullfile(cfg.outDir, 'validation_per_class.csv'));

%% ---------------- best epoch of M3 ----------------
try
    info = L3.info;
    it = info.OutputNetworkIteration;
    VH = info.ValidationHistory;
    TH = info.TrainingHistory;
    ep = TH.Epoch(find(TH.Iteration == it, 1));
    k = find(VH.Iteration == it, 1);
    fprintf('\nM3 best network: iteration %d (epoch %d of %d), validation loss %.4f, validation accuracy %.2f%%\n', ...
            it, ep, max(TH.Epoch), VH.Loss(k), VH.Accuracy(k));
catch
    fprintf('\n(best-epoch details not available in this MATLAB version)\n');
end

%% ---------------- figure 1: confusion matrices (one PNG per model) ----------------
for m = 1:3
    fig = figure('Position', [50 50 620 480], 'Color', 'w');
    try, theme(fig, 'light'); catch, end   % white figures even in MATLAB dark theme
    cc = confusionchart(fig, CM{m}, classes, 'Title', names(m), ...
        'RowSummary', 'row-normalized', 'DiagonalColor', [0.25 0.25 0.25], 'OffDiagonalColor', [0.65 0.65 0.65]);
    cc.FontName = 'Times New Roman';
    f = fullfile(cfg.outDir, sprintf('fig_confusion_M%d.png', m));
    try
        exportgraphics(cc, f, 'Resolution', 200);
    catch err
        saveas(fig, f);  fprintf('(exportgraphics failed: %s, used saveas)\n', err.message);
    end
    fprintf('Saved %s\n', f);
end

%% ---------------- figure 2: ROC curves of M3 ----------------
fig = figure('Position', [50 50 600 520], 'Color', 'w');
try, theme(fig, 'light'); catch, end   % white figures even in MATLAB dark theme
ax = axes(fig); hold(ax, 'on');
styles = {'-', '--', ':'};
ytc = categorical(cellstr(string(yt)), classes);
lg = strings(1, numel(classes));
for c = 1:numel(classes)
    [fx, fy, ~, a] = perfcurve(ytc == classes{c}, s3(:,c), true);
    plot(ax, fx, fy, 'k', 'LineStyle', styles{c}, 'LineWidth', 1.5);
    lg(c) = sprintf('%s (AUC %.4f)', classes{c}, a);
end
plot(ax, [0 1], [0 1], 'Color', [0.6 0.6 0.6]);
hold(ax, 'off'); axis(ax, 'square'); box(ax, 'on'); grid(ax, 'on');
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
xlabel(ax, 'False positive rate'); ylabel(ax, 'True positive rate');
legend(ax, [lg "chance"], 'Location', 'southeast', 'TextColor', 'k', 'Color', 'w');
title(ax, 'M3 ROC curves, one class vs rest (validation)', 'Color', 'k');
exportgraphics(fig, fullfile(cfg.outDir, 'fig_roc_validation.png'));  fprintf('Saved ROC figure\n');

%% ---------------- figure 3: wrongly classified patches (M3) ----------------
wrong = find(preds{3} ~= ytc);
fprintf('\nM3 wrong patches: %d of %d\n', numel(wrong), numel(ytc));
if ~isempty(wrong)
    show = wrong(randperm(numel(wrong), min(12, numel(wrong))));
    fig = figure('Position', [50 50 1300 520], 'Color', 'w');
try, theme(fig, 'light'); catch, end   % white figures even in MATLAB dark theme
    tiledlayout(2, 6, 'TileSpacing', 'compact');
    for k = show'
        nexttile; imshow(imread(va.files{k}));
        title(sprintf('true %s\npred %s', string(ytc(k)), string(preds{3}(k))), 'FontSize', 9, 'Color', 'k');
    end
    exportgraphics(fig, fullfile(cfg.outDir, 'fig_wrong_examples.png'));  fprintf('Saved wrong-examples figure\n');
end

fprintf('Stage 4 done. Results saved in %s\n', cfg.outDir);
