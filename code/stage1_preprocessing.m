%% STAGE 1 - PREPROCESSING
% Run step1_extract_patches.m first (data preparation: cuts labelled
% 96x96 patches and splits them train/val/test by patient).
%
% This stage applies and compares the preprocessing methods:
%   raw       - no change (baseline)
%   CLAHE     - contrast-limited adaptive histogram equalisation on the
%               L (lightness) channel, sharpens parasite / background contrast
%   Reinhard  - stain colour normalisation: matches each patch's Lab colour
%               mean and spread to a reference taken from TRAINING patches
%               only (no test information leaks in)
% No rotation is applied anywhere (dataset ReadMe rule).
%
% Part A prints and saves a dataset overview (patients, photos, annotations,
% train/val/test division). Part B does the preprocessing.
%
% Outputs (in cfg.outRoot):
%   dataset_overview.csv, per_patient_counts.csv, split_summary.csv
%   fig_dataset_overview.png                  counts as charts
%   patches_clahe\..., patches_reinhard\...   preprocessed copies, same split
%   stain_reference.mat                       reference used by later stages
%   fig_preprocessing_examples.png            raw vs CLAHE vs Reinhard

clear; clc; rng(42);

cfg.fullData = true;   % true = full dataset run, false = 1,300-patch sample
if cfg.fullData
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\patches_full';
    cfg.outRoot  = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing_full';
else
    cfg.patchDir = 'C:\Users\kisha\PROJECTS\dsp\patches_subset';
    cfg.outRoot  = 'C:\Users\kisha\PROJECTS\dsp\stage1_preprocessing';
end
if ~isfolder(cfg.outRoot), mkdir(cfg.outRoot); end

%% ================= PART A: DATASET OVERVIEW =================
% Counts patients, photos and expert annotations in the raw dataset, then
% shows how the extracted patches are divided into train / val / test.
cfg.root = 'C:\Users\kisha\PROJECTS\dsp\dataset';
src = {
 'falciparum', fullfile(cfg.root,'facliparum'), fullfile(cfg.root,'facliparum','GT_updated'), "Parasite";
 'vivax',      fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_PvTk'), fullfile(cfg.root,'vivax','NIH-NLM-ThickBloodSmearsPV','All_annotations'), "Parasitized";
 'uninfected', fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Uninfected Patients'), fullfile(cfg.root,'uninfected','NIH-NLM-ThickBloodSmearsU','Annotations'), "";
};

overview = table(); perPatient = table();
for c = 1:size(src,1)
    cls = src{c,1}; imgRoot = src{c,2}; annRoot = src{c,3}; parLabel = src{c,4};
    pats = dir(imgRoot);
    pats = pats([pats.isdir] & ~startsWith({pats.name}, '.') & ~strcmp({pats.name}, 'GT_updated'));
    pats = {pats.name}';
    nP = numel(pats);
    nImg = zeros(nP,1); nPar = zeros(nP,1); nWbc = zeros(nP,1);
    for p = 1:nP
        imgs = [dir(fullfile(imgRoot, pats{p}, '**', '*.jpg')); ...
                dir(fullfile(imgRoot, pats{p}, '**', '*.tif*'))];
        nImg(p) = numel(imgs);
        anns = dir(fullfile(annRoot, pats{p}, '*.txt'));
        for a = 1:numel(anns)
            txt = fileread(fullfile(anns(a).folder, anns(a).name));
            if strlength(parLabel) > 0
                nPar(p) = nPar(p) + count(txt, "," + parLabel + ",");
            end
            nWbc(p) = nWbc(p) + count(txt, ",White_Blood_Cell,");
        end
    end
    overview = [overview; table(string(cls), nP, sum(nImg), min(nImg), round(mean(nImg),1), max(nImg), ...
        sum(nPar), sum(nWbc), 'VariableNames', {'Class','Patients','Photos', ...
        'MinPhotosPerPatient','MeanPhotosPerPatient','MaxPhotosPerPatient', ...
        'AnnotatedParasites','AnnotatedWBCs'})]; %#ok<AGROW>
    perPatient = [perPatient; table(repmat(string(cls), nP, 1), string(pats), nImg, nPar, nWbc, ...
        'VariableNames', {'Class','Patient','Photos','Parasites','WBCs'})]; %#ok<AGROW>
end
tot = table("TOTAL", sum(overview.Patients), sum(overview.Photos), min(overview.MinPhotosPerPatient), ...
    round(sum(overview.Photos)/sum(overview.Patients),1), max(overview.MaxPhotosPerPatient), ...
    sum(overview.AnnotatedParasites), sum(overview.AnnotatedWBCs), 'VariableNames', overview.Properties.VariableNames);
overview = [overview; tot];
fprintf('\n===== RAW DATASET (thick blood smears, 4032 x 3024 photos) =====\n');
disp(overview);

% ---- how the patches are divided ----
L = strtrim(readlines(fullfile(cfg.patchDir, 'patches_index.csv')));
L(L == "") = [];
Pc = split(L(2:end), ",");            % columns: file,label,patient,split,image
T = table(Pc(:,1), Pc(:,2), Pc(:,3), Pc(:,4), Pc(:,5), ...
          'VariableNames', {'file','label','patient','split','image'});
splitSummary = table();
for s = ["train" "val" "test"]
    for cls = [string(src(:,1))' "ALL"]
        r = T.split == s;
        if cls ~= "ALL", r = r & T.label == cls; end
        splitSummary = [splitSummary; table(s, cls, numel(unique(T.patient(r))), ...
            numel(unique(T.patient(r) + "/" + T.image(r))), sum(r), round(100*sum(r)/height(T),1), ...
            'VariableNames', {'Split','Class','Patients','Photos','Patches','PercentOfPatches'})]; %#ok<AGROW>
    end
end
fprintf('\n===== TRAIN / VALIDATION / TEST DIVISION (split by patient) =====\n');
disp(splitSummary);

% ---- leakage check: no patient may appear in two splits ----
pTr = unique(T.patient(T.split=="train")); pVa = unique(T.patient(T.split=="val")); pTe = unique(T.patient(T.split=="test"));
nOverlap = numel(intersect(pTr,pVa)) + numel(intersect(pTr,pTe)) + numel(intersect(pVa,pTe));
fprintf('Patients shared between splits: %d  (must be 0)\n', nOverlap);

writetable(overview,     fullfile(cfg.outRoot, 'dataset_overview.csv'));
writetable(perPatient,   fullfile(cfg.outRoot, 'per_patient_counts.csv'));
writetable(splitSummary, fullfile(cfg.outRoot, 'split_summary.csv'));

% ---- figures ----
fig = figure('Position', [50 50 1300 380], 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact');
ov = overview(1:end-1, :);
nexttile; b = bar(categorical(ov.Class), [ov.Patients ov.Photos/10]);
legend({'Patients', 'Photos / 10'}, 'Location', 'northwest'); title('Patients and photos per class');
xtips = b(1).XEndPoints; text(xtips, b(1).YEndPoints, string(ov.Patients), 'HorizontalAlignment','center','VerticalAlignment','bottom');
text(b(2).XEndPoints, b(2).YEndPoints, string(ov.Photos), 'HorizontalAlignment','center','VerticalAlignment','bottom');
nexttile; boxchart(categorical(perPatient.Class), perPatient.Photos);
ylabel('Photos per patient'); title('Photos per patient');
nexttile;
ss = splitSummary(splitSummary.Class ~= "ALL", :);
M = reshape(ss.Patches, 3, 3)';          % rows = split, cols = class
bar(categorical(["train" "val" "test"], ["train" "val" "test"]), M, 'stacked');
legend(string(src(:,1)), 'Location', 'northeast'); ylabel('Patches'); title('Patches per split');
exportgraphics(fig, fullfile(cfg.outRoot, 'fig_dataset_overview.png'));

%% ================= PART B: PREPROCESSING =================
%% ---------------- load patches ----------------
splits = ["train" "val" "test"];
D = struct();
for s = splits
    ds = imageDatastore(fullfile(cfg.patchDir, s), 'IncludeSubfolders', true, ...
                        'LabelSource', 'foldernames');
    D.(s) = ds;
    fprintf('%-5s: %d patches\n', s, numel(ds.Files));
end
classes = categories(D.train.Labels);

%% ---------------- stain reference from training patches only ----------------
% (for the full run a random 3,000 training patches is enough for stable statistics)
pick = randperm(numel(D.train.Files), min(3000, numel(D.train.Files)));
Xtr = cellfun(@imread, D.train.Files(pick), 'UniformOutput', false);
lab = cellfun(@(P) reshape(rgb2lab(P), [], 3), Xtr, 'UniformOutput', false);
lab = vertcat(lab{:});
ref.mu = mean(lab);  ref.sd = std(lab);
save(fullfile(cfg.outRoot, 'stain_reference.mat'), 'ref');
fprintf('Stain reference (Lab mean): L=%.1f a=%.1f b=%.1f\n', ref.mu);

if cfg.fullData   % Reinhard was selected, so only it is written for the full data
    methods = {'reinhard', @(P) reinhard(P, ref)};
else
    methods = {'clahe', @claheL; 'reinhard', @(P) reinhard(P, ref)};
end

%% ---------------- write preprocessed copies ----------------
for m = 1:size(methods, 1)
    name = methods{m,1}; fn = methods{m,2};
    for s = splits
        ds = D.(s);
        for k = 1:numel(ds.Files)
            [~, f, e] = fileparts(ds.Files{k});
            out = fullfile(cfg.outRoot, ['patches_' name], s, char(ds.Labels(k)));
            if ~isfolder(out), mkdir(out); end
            imwrite(fn(imread(ds.Files{k})), fullfile(out, [f e]));
        end
    end
    fprintf('Wrote %s patches\n', name);
end

%% ---------------- figure 1: examples ----------------
nEx = 4;
fig = figure('Position', [50 50 1300 700], 'Color', 'w');
t = tiledlayout(3, numel(classes)*nEx, 'TileSpacing', 'tight', 'Padding', 'compact');
rowNames = {'Raw', 'CLAHE', 'Reinhard'};
fns = {@(P) P, @claheL, @(P) reinhard(P, ref)};
for r = 1:3
    for c = 1:numel(classes)
        idx = find(D.test.Labels == classes{c});
        idx = idx(round(linspace(1, numel(idx), nEx)));
        for k = idx'
            nexttile;
            imshow(fns{r}(imread(D.test.Files{k})));
            if r == 1 && k == idx(1), title(classes{c}); end
            if c == 1 && k == idx(1), ylabel(rowNames{r}, 'Visible', 'on', 'FontWeight', 'bold'); end
        end
    end
end
title(t, 'Preprocessing comparison (rows: raw / CLAHE / Reinhard; test patients)');
exportgraphics(fig, fullfile(cfg.outRoot, 'fig_preprocessing_examples.png'));

fprintf('Stage 1 done. Outputs in %s\n', cfg.outRoot);

%% ======================= helpers =======================
function P = claheL(P)
    lab = rgb2lab(P);
    lab(:,:,1) = 100 * adapthisteq(lab(:,:,1)/100, 'ClipLimit', 0.01, 'NumTiles', [4 4]);
    P = im2uint8(lab2rgb(lab));
end

function P = reinhard(P, ref)
    lab = rgb2lab(P);
    for ch = 1:3
        v = lab(:,:,ch);
        lab(:,:,ch) = (v - mean(v(:))) / (std(v(:)) + eps) * ref.sd(ch) + ref.mu(ch);
    end
    P = im2uint8(lab2rgb(lab));
end
