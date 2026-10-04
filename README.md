# Prediction of Human Oral Bioavailability with Machine Learning, Deep Learning and SHAP

MSc thesis project: *Prediction of Human Oral Bioavailability Using Machine and Deep Learning and Model Interpretation with SHAP Values*.

**Author:** Marina Tsiantouka

Human oral bioavailability (HOB) is the fraction of an orally administered dose that reaches the systemic circulation unchanged. This repository contains a complete QSAR workflow that predicts HOB class from molecular structure, compares 2D and 3D molecular descriptors, and explains the predictions with SHAP values.

## Overview

- **Task:** binary classification (High / Low HOB) under two label definitions, a 50% cutoff (nearly balanced) and a 20% cutoff (imbalanced).
- **Input:** 1,826 Mordred descriptors (1,613 2D and 213 3D) computed from SMILES.
- **Models:** tree ensembles (ExtraTrees, CatBoost) as final models, and a multilayer perceptron for comparison.
- **Evaluation:** all decisions are made with 5-fold cross-validation on the training set. Two external test sets are used once, at the very end.
- **Interpretation:** SHAP values for both the tree models and the neural network, including the share of 2D versus 3D descriptors.

## Dataset

The data come from HobPre (Wei et al., 2022). Each molecule is given as a SMILES string with its experimental HOB value and two binary labels.

| Subset | Molecules | Cutoff 50% (High / Low) | Cutoff 20% (High / Low) |
|---|---|---|---|
| Training set | 1,142 | 621 / 521 | 859 / 283 |
| Test set 1 | 290 (287 at 20%) | 169 / 121 | 214 / 73 |
| Test set 2 | 140 (132 at 20%) | 89 / 51 | 127 / 5 |

Test set 2 has only five Low molecules at the 20% cutoff, so class-specific metrics for that case rest on five observations and should be read as a stress test.

## Workflow

### 1. Descriptors and missing values — `preprocess_data_new.ipynb`

- 3D conformers are generated with RDKit (ETKDGv3) and optimised with MMFF. Mordred then computes all 1,826 descriptors.
- Missing values are handled by how often they occur:
  - descriptors missing in every molecule are dropped,
  - descriptors missing in more than 20% of molecules are kept (filled with 0) only if their point-biserial correlation with the label satisfies |r| > 0.10 and p < 0.05,
  - atom-type descriptors missing in 5–20% of molecules are filled with 0, after checking that the missing value really means the atom type is absent,
  - descriptors missing in fewer than 5% of molecules are imputed with the training median.
- This leaves **1,672 descriptors** with no missing values.

### 2. Feature selection

Two methods are applied separately for each cutoff and then compared.

**MI ∩ CatBoost — `feature_selection_243.ipynb`**

1. Variance threshold (0.01).
2. Correlation filter: for every pair with |r| > 0.95, the descriptor less correlated with the label is removed. About 720 descriptors remain.
3. Ranking by mutual information and, independently, by CatBoost feature importance.
4. The final set is the intersection of the two top-400 lists.

**Elastic Net — `feature_selection_elasticnet.ipynb`**

Logistic regression with an elastic-net penalty (SAGA solver, balanced class weights), fitted on the same filtered pool over a grid of `C` and `l1_ratio`. The selected descriptors are those with non-zero coefficients.

| Cutoff | MI ∩ CatBoost | of which 2D / 3D | Elastic Net |
|---|---|---|---|
| 50% | 243 | 195 / 48 | 318 |
| 20% | 232 | 183 / 49 | 346 |

### 3. Model selection — `hyperparameter_tunning_new.ipynb`

Every model is a pipeline of `RobustScaler` and a classifier, scored by ROC-AUC with stratified 5-fold cross-validation (seed 42). The notebook runs four stages and logs every experiment:

1. **Grid search** for Logistic Regression, Random Forest and CatBoost.
2. **Out-of-fold predictions** for seven classifiers: Logistic Regression, Random Forest, ExtraTrees, CatBoost, SVM (RBF kernel), k-Nearest Neighbours and Gaussian Naive Bayes.
3. **Soft-average ensemble search** over every combination of those seven models.
4. **Tuned soft-voting ensembles:** CatBoost + XGBoost, and Random Forest + ExtraTrees + CatBoost.

Selected models (MI ∩ CatBoost feature set, out-of-fold ROC-AUC of about 0.77 in both scenarios):

| Cutoff | Final model |
|---|---|
| 50% | ExtraTrees + CatBoost, soft average |
| 20% | ExtraTrees |

ExtraTrees uses 300 trees, `max_features='log2'`, `min_samples_split=5` and balanced class weights. CatBoost uses depth 4, learning rate 0.15, 300 iterations and balanced class weights.

### 4. External evaluation — `test_set_evaluation_final.ipynb`

The decision threshold is chosen on the out-of-fold training predictions by maximising Youden's J (0.572 for the 50% cutoff, 0.713 for the 20% cutoff). The model is then refitted on all 1,142 training molecules and applied unchanged to the test sets.

| Cutoff | Model | Test set | n | ROC-AUC | MCC | Balanced acc. | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 50% | ExtraTrees + CatBoost | 1 | 290 | 0.820 | 0.478 | 0.742 | 0.761 | 0.716 | 0.769 |
| 50% | ExtraTrees + CatBoost | 2 | 140 | 0.895 | 0.629 | 0.822 | 0.854 | 0.820 | 0.824 |
| 20% | ExtraTrees | 1 | 287 | 0.830 | 0.456 | 0.755 | 0.815 | 0.743 | 0.767 |
| 20% | ExtraTrees | 2 | 132 | 0.981 | 0.366 | 0.902 | 0.891 | 0.803 | 1.000 |

### 5. Comparison of the two feature sets — `test_comparison_feature_selection.ipynb`

The same models, protocol and molecules are used, and only the descriptor set changes. Results for ExtraTrees, Random Forest, CatBoost and Logistic Regression on both test sets are in `results/Feature_selection_comparison/`.

### 6. 2D versus 3D descriptors — `comparison_2d_vs_3d.ipynb`

Cross-validation on the training set only, with the final model of each scenario.

- **Experiment A:** the selected feature set is split into its 2D and 3D parts, and the model is trained on 2D only, 3D only, and both.
- **Experiment B:** the top-*k* 2D and top-*k* 3D descriptors by mutual information are compared at equal size.

| Cutoff | 2D only | 3D only | 2D + 3D |
|---|---|---|---|
| 50% | 0.773 (195) | 0.726 (48) | 0.770 (243) |
| 20% | 0.778 (183) | 0.738 (49) | 0.775 (232) |

Values are cross-validated ROC-AUC from Experiment A, with the number of descriptors in parentheses.

### 7. Neural network — `neural_network.ipynb`

A multilayer perceptron (Keras 3 / TensorFlow) trained on all 1,672 descriptors.

- Each hidden layer is `Dense → LayerNormalization → ReLU → Dropout`, with a sigmoid output.
- Adam optimiser (learning rate 3e-4), binary cross-entropy, balanced class weights, and early stopping on validation ROC-AUC.
- Grid of 27 configurations (1–3 hidden layers × 64/128/256 units × dropout 0.2/0.35/0.5), scored with the same 5 folds.
- The final prediction is the average of three networks trained with different seeds.

| Cutoff | Best configuration | Test set 1 ROC-AUC | Test set 2 ROC-AUC |
|---|---|---|---|
| 50% | 1 × 256, dropout 0.5 | 0.794 | 0.879 |
| 20% | 1 × 64, dropout 0.2 | 0.779 | 0.943 |

### 8. Interpretation with SHAP — `shap_analysis.ipynb`, `neural_network_shap.ipynb`

- **Tree models:** `TreeExplainer`, giving exact SHAP values in probability units. Outputs include global importance (bar and beeswarm plots), dependence plots, and waterfall plots for individual test molecules.
- **Neural network:** `GradientExplainer` (expected gradients), averaged over the three networks.
- **2D versus 3D:** 3D descriptors are about 20% of the selected set and account for about 22–24% of the total mean |SHAP|.

## Repository structure

```
notebooks/   Jupyter notebooks, one per stage of the workflow
Data/        HobPre dataset, computed descriptors and selected feature sets
results/     Tables and figures, grouped by label cutoff
models/      Saved neural networks and their preprocessing
```

Notebooks in the order they are run:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `preprocess_data_new.ipynb` | Descriptor calculation and missing-value handling |
| 2 | `feature_selection_243.ipynb` | MI ∩ CatBoost feature selection |
| 3 | `feature_selection_elasticnet.ipynb` | Elastic Net feature selection |
| 4 | `hyperparameter_tunning_new.ipynb` | Model screening, tuning and ensemble search |
| 5 | `test_set_evaluation_final.ipynb` | Threshold selection and external evaluation |
| 6 | `test_comparison_feature_selection.ipynb` | MI ∩ CatBoost versus Elastic Net on the test sets |
| 7 | `comparison_2d_vs_3d.ipynb` | Predictive power of 2D and 3D descriptors |
| 8 | `neural_network.ipynb` | MLP tuning, training and evaluation |
| 9 | `shap_analysis.ipynb` | SHAP for the tree models |
| 10 | `neural_network_shap.ipynb` | SHAP for the neural network |

## Running the code

The notebooks were developed with Python 3.10.

```bash
pip install numpy pandas scipy scikit-learn catboost xgboost rdkit mordred shap tensorflow matplotlib openpyxl jupyter
```

Each notebook has its settings in the first code cell:

- `label_case`: `'label_cutoff_50%'` or `'label_cutoff_20%'`
- `fs_method` (tuning notebook): `'mi_catboost'` or `'elasticnet'`
- `test_set` (evaluation notebook): `'Test set 1'` or `'Test set 2'`

Run a notebook once per setting to reproduce both scenarios. Paths are relative, so start Jupyter from the `notebooks/` folder. The random seed is fixed at 42 throughout.

Code comments and some figure labels are in Greek.

## References

- Wei, M. et al. HobPre: accurate prediction of human oral bioavailability for small molecules. *Journal of Cheminformatics* 14, 1 (2022).
- Moriwaki, H. et al. Mordred: a molecular descriptor calculator. *Journal of Cheminformatics* 10, 4 (2018).
- Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. *NeurIPS* (2017).
