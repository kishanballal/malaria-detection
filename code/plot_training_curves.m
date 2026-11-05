%% Re-draws the M3 training curves on a white background (no retraining needed)
% Reads the training history saved by stage3_training.m and writes
% fig_training_curves_white.png into the same folder.

clear; clc;
modelDir = 'C:\Users\kisha\PROJECTS\dsp\stage3_models_full';
L = load(fullfile(modelDir, 'M3_efficientnetb0_finetuned.mat'), 'info');
TH = L.info.TrainingHistory;  VH = L.info.ValidationHistory;

% x axis in epochs instead of iterations
if ismember('Epoch', TH.Properties.VariableNames)
    itPerEpoch = max(TH.Iteration) / max(TH.Epoch);
else   % validation ran once per epoch, so its spacing is one epoch
    itPerEpoch = median(diff(VH.Iteration));
    TH.Epoch = ceil(TH.Iteration / itPerEpoch);
end
tx = TH.Iteration / itPerEpoch;   vx = VH.Iteration / itPerEpoch;
% training curve averaged per epoch (raw per-iteration values are very noisy)
ep = unique(TH.Epoch);
trAcc  = arrayfun(@(e) mean(TH.Accuracy(TH.Epoch == e)), ep);
trLoss = arrayfun(@(e) mean(TH.Loss(TH.Epoch == e)), ep);
best = NaN;
try, best = L.info.OutputNetworkIteration / itPerEpoch; catch, end

fig = figure('Position', [50 50 1000 700], 'Color', 'w');
try, theme(fig, 'light'); catch, end          % ignore MATLAB dark theme
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact');

ax1 = nexttile(tl);
plot(ax1, tx, TH.Accuracy, 'Color', [0.82 0.82 0.82]); hold(ax1, 'on');
h1 = plot(ax1, ep, trAcc, 'k--', 'LineWidth', 1.5);
h2 = plot(ax1, vx, VH.Accuracy, 'k-o', 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', 'k');
if ~isnan(best), xline(ax1, best, 'k:', 'best epoch', 'LineWidth', 1.2, 'LabelVerticalAlignment', 'bottom'); end
hold(ax1, 'off'); ylim(ax1, [40 100.5]); xlim(ax1, [0 max(ep)]);
ylabel(ax1, 'Accuracy (%)'); grid(ax1, 'on'); box(ax1, 'on');
legend(ax1, [h1 h2], {'Training (mean per epoch)', 'Validation'}, 'Location', 'southeast');
title(ax1, 'EfficientNet-b0 fine-tuning');

ax2 = nexttile(tl);
plot(ax2, tx, TH.Loss, 'Color', [0.82 0.82 0.82]); hold(ax2, 'on');
h1 = plot(ax2, ep, trLoss, 'k--', 'LineWidth', 1.5);
h2 = plot(ax2, vx, VH.Loss, 'k-o', 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', 'k');
if ~isnan(best), xline(ax2, best, 'k:', 'best epoch', 'LineWidth', 1.2); end
hold(ax2, 'off'); xlim(ax2, [0 max(ep)]);
ylabel(ax2, 'Cross-entropy loss'); xlabel(ax2, 'Epoch'); grid(ax2, 'on'); box(ax2, 'on');
legend(ax2, [h1 h2], {'Training (mean per epoch)', 'Validation'}, 'Location', 'northeast');

for ax = [ax1 ax2]
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.6 0.6 0.6], 'FontName', 'Times New Roman', 'FontSize', 12);
    ax.Title.Color = 'k';  ax.Legend.TextColor = 'k';  ax.Legend.Color = 'w';  ax.Legend.EdgeColor = 'k';
end
exportgraphics(fig, fullfile(modelDir, 'fig_training_curves_white.png'), 'BackgroundColor', 'white', 'Resolution', 200);
fprintf('Saved %s\n', fullfile(modelDir, 'fig_training_curves_white.png'));
