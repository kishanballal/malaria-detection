# Malaria detection from thick blood smears (MATLAB)

Three-class classifier for thick-smear blood images: **P. falciparum**, **P. vivax** or **uninfected**.
Built in MATLAB on the NIH/NLM thick-smear datasets, using Reinhard colour normalisation and a fine-tuned EfficientNet-b0.

Base paper: Mujahid et al. 2024, *Efficient deep learning-based approach for malaria detection using red blood cell smears* (Sci. Rep.), EfficientNet on binary thin-smear cells. This project extends the idea to thick smears and three classes.

## Data

NIH/NLM thick-smear sets (not included in this repo, about 10 GB):

| Folder | Source | Content |
|---|---|---|
| `dataset/facliparum` | NLM-Falciparum-Thick-150Patients | 1,883 photos, 84.5k annotated parasites |
| `dataset/vivax` | NIH-NLM-ThickBloodSmearsPV | 3,014 photos, 43k annotated parasites |
| `dataset/uninfected` | NIH-NLM-ThickBloodSmearsU | 1,141 photos, 27k annotated WBCs |

Photos are 4032x3024. Images are never rotated (the dataset ReadMe says annotation coordinates depend on the original orientation).
All splits are **by patient** (about 70/15/15) to avoid leakage. 63,713 patches were extracted from the full set (train 44,587, val 9,860, test 9,266).

## Pipeline and results

Scripts are in `code/`. Run them in order from the project root; each stage writes figures and CSVs to its own `stage*_full` folder.

| Stage | Script | What it does | Result |
|---|---|---|---|
| 0 | `step1_extract_patches.m` | Cut 96 px patches around annotated parasites / WBCs, patient split | 63,713 patches |
| 1 | `stage1_preprocessing.m` | Reinhard colour normalisation (reference from 3,000 train patches) | Normalised patch set |
| 2 | `stage2_feature_extraction.m` | Colour, GLCM, HOG, LBP and deep (EfficientNet) features, silhouette + t-SNE | Silhouette: LBP 0.127, deep 0.083, handcrafted -0.019 |
| 3 | `stage3_training.m` | M1 SVM on handcrafted, M2 SVM on deep features, M3 fine-tuned EfficientNet-b0 (35 epochs) | Best M3 checkpoint at epoch 12 |
| 4 | `stage4_validation.m` | Validation metrics, patient-level vote, confusion, ROC | M1 92.66%, M2 93.96%, **M3 99.81% / F1 99.79 / AUC 0.9999** (54 patients) |
| 5 | `stage5_testing.m` | Held-out test with Wilson 95% CI | M1 91.46%, M2 91.72%, **M3 99.44% (CI 99.26-99.57) / F1 99.41 / AUC 0.9998**, 51/51 patients correct, 2.56 ms/patch |
| 6 | `stage6_whole_photo.m` | Candidate blob detection on whole photos, then M3 | 98.0% photos, 100% patients, parasite recall 95.5%, **precision 12.4%** (see known issues) |
| 7 | `stage7_gradcam.m` | Grad-CAM on test patches | 42.5% of heat in the patch centre (chance 19.6%); peak on the parasite in 83% of patches |

Other scripts: `classify_photo.m` (pick a photo, get a diagnosis) and `plot_training_curves.m`.

An earlier pilot on 1,300 patches compared six pipelines; fine-tuned EfficientNet-b0 came out best (95.2%), which is why it is used here.

The final report will be added here once it is ready.

## Known issues

- **Whole-photo detection (stage 6) has low precision.** On infected photos the model labels many WBCs and debris as parasites. The "uninfected" class was trained only on WBC/background patches from uninfected patients, so the model partly learned which photo/source a patch came from. Planned fix: add a "not a parasite" class with WBC and background patches from *all* photos (hard negatives) and rerun stages 1-5.
- **Source confound.** Vivax photos come from a different collection than falciparum and uninfected, so part of the very high patch accuracy may reflect source differences rather than species alone.

## Not in this repo

Excluded via `.gitignore` (kept only on the training laptop):

- `dataset/` raw photos and annotations (~10 GB) and the extracted patches (`patches_full/`, `stage1_preprocessing_full/patches_*`)
- `stage3_models_full/M1_svm_handcrafted.mat` (123 MB) and `M2_svm_deep.mat` (159 MB), over GitHub's 100 MB limit
- `stage2_features_full/features_*.mat` (650 + 144 + 135 MB)

The final model, `stage3_models_full/M3_efficientnetb0_finetuned.mat` (16 MB), is included.

## Requirements

MATLAB with Deep Learning, Image Processing, Statistics and Machine Learning, and Computer Vision toolboxes, plus the EfficientNet-b0 support package. Trained on an RTX 5060 8 GB (stage 3 took about 3 h 40 min).
