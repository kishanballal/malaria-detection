%% STAGE 5 - TESTING (final, unseen patients)
% Evaluates the three models from stage 3 on the TEST patients
% (51 patients never used for training, model choice or epoch choice).
%   M1  SVM on handcrafted features
%   M2  SVM on EfficientNet-b0 deep features
%   M3  EfficientNet-b0 fine-tuned (main model)
% Reports patch-level accuracy with 95% confidence interval, precision,
% recall, F1, AUC, patient-level accuracy, a per-patient table for M3,
% inference speed, and a validation vs test comparison.

clear; clc; rng(42);

cfg.fullData = true;   % true = full dataset run, false = 1,300-patch sample
if cfg.fullData
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features_full';
    cfg.modelDir = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full';
    cfg.valDir   = 'C:\Users\kisha\PROJECTS\dsp\stage4_validation_full';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage5_testing_full';
else
    cfg.featDir  = 'C:\Users\kisha\PROJECTS\dsp\stage2_features';
    cfg.modelDir = 'C:\Users\kisha\PROJECTS\dsp\stage3_models';
    cfg.valDir   = 'C:\Users\kisha\PROJECTS\dsp\stage4_validation';
    cfg.outDir   = 'C:\Users\kisha\PROJECTS\dsp\stage5_testing';
end
cfg.batch = 32;
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

%% ---------------- load models and test data ----------------
fprintf('Loading models and test features...\n');
te = load(fullfile(cfg.featDir, 'features_test.mat'));
L1 = load(fullfile(cfg.modelDir, 'M1_svm_handcrafted.mat'));
L2 = load(fullfile(cfg.modelDir, 'M2_svm_deep.mat'));
L3 = load(fullfile(cfg.modelDir, 'M3_efficientnetb0_finetuned.mat'));
classes = L3.classes(:)';
yt  = te.Y(:);
ytc = categorical(cellstr(string(yt)), classes);
nP  = numel(unique(te.patient));
fprintf('Test: %d patches from %d patients\n', numel(yt), nP);

%% ---------------- predictions (timed) ----------------
fprintf('[1/3] M1 predicting...\n');
t0 = tic; [yp1, s1] = predict(L1.M1, te.Fhand); tm(1) = toc(t0);
fprintf('[2/3] M2 predicting...\n');
t0 = tic; [yp2, s2] = predict(L2.M2, te.Fdeep); tm(2) = toc(t0);
fprintf('[3/3] M3 predicting on %d patches (GPU)...\n', numel(yt));
inSz = L3.M3.Layers(1).InputSize(1:2);
ds = imageDatastore(te.files);                     % same order as the feature file
t0 = tic;
s3 = minibatchpredict(L3.M3, augmentedImageDatastore(inSz, ds), 'MiniBatchSize', cfg.batch);
tm(3) = toc(t0);
yp3 = scores2label(s3, classes);
% note: M1/M2 times exclude feature extraction (stage 2); M3 time includes reading the images

names  = ["M1 SVM, handcrafted features"; "M2 SVM, EfficientNet-b0 features"; "M3 EfficientNet-b0 fine-tuned"];
preds  = {yp1(:), yp2(:), yp3(:)};
scores = {s1, s2, s3};

%% ---------------- metrics ----------------
summary = table();  perClass = table();  CM = cell(1, 3);
[pats, ~, g] = unique(te.patient(:));
for m = 1:3
    yp = categorical(cellstr(string(preds{m})), classes);
    C = confusionmat(ytc, yp, 'Order', categorical(classes));
    prec = diag(C) ./ max(sum(C,1)', 1);
    rec  = diag(C) ./ max(sum(C,2), 1);
    f1   = 2*prec.*rec ./ max(prec+rec, eps);
    auc  = zeros(numel(classes), 1);
    for c = 1:numel(classes)
        [~, ~, ~, auc(c)] = perfcurve(ytc == classes{c}, scores{m}(:,c), true);
    end
    acc = mean(yp == ytc);
    [lo, hi] = wilson(sum(yp == ytc), numel(ytc));       % 95% confidence interval
    % patient level: a patient is labelled with the majority vote of its patches
    pt = splitapply(@(x) mode(x), ytc, g);
    pp = splitapply(@(x) mode(x), yp,  g);
    summary = [summary; table(names(m), round(100*acc, 2), sprintf("%.2f - %.2f", 100*lo, 100*hi), ...
        round(100*mean(f1), 2), round(mean(auc), 4), sprintf("%d / %d", sum(pp == pt), numel(pats)), ...
        round(100*mean(pp == pt), 1), round(1000*tm(m)/numel(ytc), 3), ...
        'VariableNames', {'Model','PatchAccuracy','CI95','MacroF1','MeanAUC','PatientsCorrect','PatientAccuracy','msPerPatch'})]; %#ok<AGROW>
    perClass = [perClass; table(repmat(names(m), numel(classes), 1), string(classes(:)), sum(C,2), ...
        round(100*prec, 2), round(100*rec, 2), round(100*f1, 2), round(auc, 4), ...
        'VariableNames', {'Model','Class','Patches','Precision','Recall','F1','AUC'})]; %#ok<AGROW>
    CM{m} = C;
    preds{m} = yp;
end

fprintf('\n===== TEST RESULTS (%d patches, %d unseen patients) =====\n', numel(yt), nP);
disp(summary);
fprintf('\n===== PER CLASS =====\n');
disp(perClass);
writetable(summary,  fullfile(cfg.outDir, 'test_summary.csv'));
writetable(perClass, fullfile(cfg.outDir, 'test_per_class.csv'));

%% ---------------- per-patient table for M3 ----------------
yp = preds{3};
perPatient = table(string(pats), string(splitapply(@(x) mode(x), ytc, g)), ...
    string(splitapply(@(x) mode(x), yp, g)), accumarray(g, 1), ...
    round(100*accumarray(g, double(yp == ytc)) ./ accumarray(g, 1), 1), ...
    'VariableNames', {'Patient','TrueClass','PredictedClass','Patches','PatchesCorrectPercent'});
perPatient = sortrows(perPatient, 'PatchesCorrectPercent');
writetable(perPatient, fullfile(cfg.outDir, 'test_per_patient_M3.csv'));
fprintf('\n===== M3: 10 HARDEST TEST PATIENTS (all %d saved in test_per_patient_M3.csv) =====\n', nP);
disp(perPatient(1:min(10, height(perPatient)), :));

%% ---------------- validation vs test ----------------
try
    V = readtable(fullfile(cfg.valDir, 'validation_summary.csv'), 'TextType', 'string');
    cmp = table(names, V.PatchAccuracy, summary.PatchAccuracy, V.MacroF1, summary.MacroF1, ...
        'VariableNames', {'Model','ValAccuracy','TestAccuracy','ValMacroF1','TestMacroF1'});
    fprintf('\n===== VALIDATION vs TEST =====\n');
    disp(cmp);
    writetable(cmp, fullfile(cfg.outDir, 'validation_vs_test.csv'));
catch
    fprintf('\n(validation_summary.csv not found, run stage 4 first for the comparison)\n');
end

%% ---------------- figure 1: confusion matrices (one PNG per model) ----------------
for m = 1:3
    fig = figure('Position', [50 50 620 480], 'Color', 'w');
    try, theme(fig, 'light'); catch, end   % white figures even in MATLAB dark theme
    cc = confusionchart(fig, CM{m}, classes, 'Title', names(m) + " (test)", ...
        'RowSummary', 'row-normalized', 'DiagonalColor', [0.25 0.25 0.25], 'OffDiagonalColor', [0.65 0.65 0.65]);
    cc.FontName = 'Times New Roman';
    f = fullfile(cfg.outDir, sprintf('fig_confusion_test_M%d.png', m));
    try
        exportgraphics(cc, f, 'Resolution', 200);
    catch err
        saveas(fig, f);  fprintf('(exportgraphics failed: %s, used saveas)\n', err.message);
    end
    fprintf('Saved %s\n', f);
end

%% ---------------- figure 2: ROC curves, all three models ----------------
fig = figure('Position', [50 50 1500 480], 'Color', 'w');
try, theme(fig, 'light'); catch, end
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact');
styles = {'-', '--', ':'};
for m = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    lg = strings(1, numel(classes));
    for c = 1:numel(classes)
        [fx, fy, ~, a] = perfcurve(ytc == classes{c}, scores{m}(:,c), true);
        plot(ax, fx, fy, 'k', 'LineStyle', styles{c}, 'LineWidth', 1.5);
        lg(c) = sprintf('%s (AUC %.4f)', classes{c}, a);
    end
    plot(ax, [0 1], [0 1], 'Color', [0.6 0.6 0.6]);
    hold(ax, 'off'); axis(ax, 'square'); box(ax, 'on'); grid(ax, 'on');
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    xlabel(ax, 'False positive rate'); ylabel(ax, 'True positive rate');
    legend(ax, [lg "chance"], 'Location', 'southeast', 'TextColor', 'k', 'Color', 'w');
    title(ax, names(m), 'Color', 'k');
end
exportgraphics(fig, fullfile(cfg.outDir, 'fig_roc_test.png'));  fprintf('Saved ROC figure\n');

%% ---------------- figure 3: wrongly classified patches (M3) ----------------
wrong = find(preds{3} ~= ytc);
fprintf('\nM3 wrong patches: %d of %d\n', numel(wrong), numel(ytc));
if ~isempty(wrong)
    show = wrong(randperm(numel(wrong), min(12, numel(wrong))));
    fig = figure('Position', [50 50 1300 520], 'Color', 'w');
    try, theme(fig, 'light'); catch, end
    tiledlayout(fig, 2, 6, 'TileSpacing', 'compact');
    for k = show'
        nexttile; imshow(imread(te.files{k}));
        title(sprintf('true %s\npred %s', string(ytc(k)), string(preds{3}(k))), 'FontSize', 9, 'Color', 'k');
    end
    exportgraphics(fig, fullfile(cfg.outDir, 'fig_wrong_examples_test.png'));  fprintf('Saved wrong-examples figure\n');
end

fprintf('Stage 5 done. Results saved in %s\n', cfg.outDir);

%% ======================= helpers =======================
function [lo, hi] = wilson(k, n)
% 95% Wilson score interval for a proportion k/n
    z = 1.96;  p = k/n;
    c = (p + z^2/(2*n)) / (1 + z^2/n);
    h = z*sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n);
    lo = c - h;  hi = c + h;
end
