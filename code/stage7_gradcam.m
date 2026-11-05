%% STAGE 7 - GRAD-CAM: WHERE DOES THE MODEL LOOK?
% Grad-CAM heatmaps of the fine-tuned EfficientNet-b0 (M3) on TEST patches.
% Red = regions that pushed the model towards its predicted class.
%   Figure 1: correctly classified patches, 4 per class
%   Figure 2: wrongly classified patches
%   Check   : test patches are cut centred on the expert-annotated parasite,
%             so if the model looks at the parasite, most of the heat should
%             lie in the centre of the patch. We measure this on random
%             correctly classified parasite patches and compare with chance.

clear; clc; rng(42);

cfg.featDir   = 'C:\Users\kisha\PROJECTS\dsp\stage2_features_full';
cfg.modelFile = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full\M3_efficientnetb0_finetuned.mat';
cfg.outDir    = 'C:\Users\kisha\PROJECTS\dsp\stage7_gradcam_full';
cfg.nCheck    = 300;     % parasite patches used for the centre check
cfg.centreR   = 24;      % centre circle radius in the 96 x 96 patch (pixels)
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

L = load(cfg.modelFile);  net = L.M3;  classes = L.classes(:)';
inSz = net.Layers(1).InputSize(1:2);
te = load(fullfile(cfg.featDir, 'features_test.mat'), 'files', 'Y');
yt = categorical(cellstr(string(te.Y(:))), classes);

fprintf('Predicting %d test patches...\n', numel(yt));
sc = minibatchpredict(net, augmentedImageDatastore(inSz, imageDatastore(te.files)), 'MiniBatchSize', 64);
yp = scores2label(sc, classes);  yp = categorical(cellstr(string(yp(:))), classes);
conf = max(sc, [], 2);

%% ---------------- figure 1: correct patches, 4 per class ----------------
nEx = 4;
fig = figure('Position', [30 30 1500 620], 'Color', 'w');
try, theme(fig, 'light'); catch, end
tl = tiledlayout(fig, numel(classes), 2*nEx, 'TileSpacing', 'tight', 'Padding', 'compact');
for c = 1:numel(classes)
    idx = find(yt == classes{c} & yp == yt);
    idx = idx(randperm(numel(idx), nEx));
    for k = idx'
        [I, M] = camOf(net, te.files{k}, inSz, find(yp(k) == classes));
        nexttile(tl); imshow(I); title(sprintf('%s', classes{c}), 'FontSize', 9, 'Color', 'k');
        nexttile(tl); showCam(I, M); title(sprintf('Grad-CAM (%.0f%%)', 100*conf(k)), 'FontSize', 9, 'Color', 'k');
    end
end
title(tl, 'Grad-CAM of M3 on correctly classified test patches (red = most important)', 'FontSize', 11);
exportgraphics(fig, fullfile(cfg.outDir, 'fig_gradcam_correct.png'), 'Resolution', 150);
fprintf('Saved fig_gradcam_correct.png\n');

%% ---------------- figure 2: wrong patches ----------------
wrong = find(yp ~= yt);
fprintf('Wrong test patches: %d\n', numel(wrong));
if ~isempty(wrong)
    show = wrong(randperm(numel(wrong), min(8, numel(wrong))));
    fig = figure('Position', [30 30 1500 420], 'Color', 'w');
    try, theme(fig, 'light'); catch, end
    tl = tiledlayout(fig, 2, numel(show), 'TileSpacing', 'tight', 'Padding', 'compact');
    for j = 1:numel(show)
        k = show(j);
        [I, M] = camOf(net, te.files{k}, inSz, find(yp(k) == classes));
        nexttile(tl, j); imshow(I);
        title(sprintf('true %s\npred %s', string(yt(k)), string(yp(k))), 'FontSize', 9, 'Color', 'k');
        nexttile(tl, j + numel(show)); showCam(I, M);
    end
    title(tl, 'Grad-CAM of M3 on wrongly classified test patches', 'FontSize', 11);
    exportgraphics(fig, fullfile(cfg.outDir, 'fig_gradcam_wrong.png'), 'Resolution', 150);
    fprintf('Saved fig_gradcam_wrong.png\n');
end

%% ---------------- centre check on parasite patches ----------------
isPar = (yt == 'falciparum' | yt == 'vivax') & yp == yt;
idx = find(isPar);  idx = idx(randperm(numel(idx), min(cfg.nCheck, numel(idx))));
[xx, yy] = meshgrid(1:inSz(2), 1:inSz(1));
rr = cfg.centreR * inSz(1) / 96;                        % radius at network input size
inC = hypot(xx - (inSz(2)+1)/2, yy - (inSz(1)+1)/2) <= rr;
chance = mean(inC(:));
frac = zeros(numel(idx), 1);  peakIn = false(numel(idx), 1);
t0 = tic;
for j = 1:numel(idx)
    k = idx(j);
    [~, M] = camOf(net, te.files{k}, inSz, find(yp(k) == classes));
    M = max(M, 0);
    frac(j) = sum(M(inC)) / max(sum(M(:)), eps);
    [~, pk] = max(M(:));  peakIn(j) = inC(pk);
    if mod(j, 50) == 0, fprintf('  centre check %d / %d (%.1f min)\n', j, numel(idx), toc(t0)/60); end
end
fprintf('\n===== GRAD-CAM CENTRE CHECK (%d correctly classified parasite test patches) =====\n', numel(idx));
fprintf('Heat inside the centre circle (radius %d px of 96): %.1f%% on average  (chance: %.1f%%)\n', ...
        cfg.centreR, 100*mean(frac), 100*chance);
fprintf('Hottest point inside the centre circle: %.1f%% of patches  (chance: %.1f%%)\n', 100*mean(peakIn), 100*chance);
chk = table(["Patches checked"; "Mean heat in centre circle (%)"; "Hottest point in centre circle (%)"; "Chance level (%)"], ...
            [numel(idx); 100*mean(frac); 100*mean(peakIn); 100*chance], 'VariableNames', {'Measure','Value'});
writetable(chk, fullfile(cfg.outDir, 'gradcam_centre_check.csv'));

fig = figure('Position', [50 50 600 420], 'Color', 'w');
try, theme(fig, 'light'); catch, end
ax = axes(fig);
histogram(ax, 100*frac, 0:5:100, 'FaceColor', [0.45 0.45 0.45]);
xline(ax, 100*chance, 'k--', 'chance', 'LineWidth', 1.2);
xlabel(ax, 'Share of Grad-CAM heat inside the centre circle (%)'); ylabel(ax, 'Patches');
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k'); grid(ax, 'on');
exportgraphics(fig, fullfile(cfg.outDir, 'fig_gradcam_centre_check.png'), 'Resolution', 200);

fprintf('Stage 7 done. Results saved in %s\n', cfg.outDir);

%% ======================= helpers =======================
function [I, M] = camOf(net, file, inSz, cls)
    I = imresize(imread(file), inSz);
    M = gradCAM(net, single(I), cls);
    M = rescale(M);
end

function showCam(I, M)
    imshow(I); hold on;
    h = imagesc(M); colormap(gca, jet); h.AlphaData = 0.45;
    hold off; axis image off;
end
