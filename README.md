# Credit Risk ML (scikit-learn)

## Project overview
This project builds a credit-risk classifier to predict `dlq_2yrs` (2-year delinquency) using a Kaggle credit risk benchmark dataset. Goal: compare a simple baseline (Logistic Regression) vs a stronger tabular model (HistGradientBoosting), then validate performance, interpret drivers, and check probability calibration.

## Dataset
Source: Kaggle – Credit Risk Benchmark Dataset  
File used: `Credit Risk Benchmark Dataset.csv` (11 columns)

## Approach
- Baseline: Logistic Regression (with standardization)
- Final model: HistGradientBoostingClassifier (better ROC-AUC and Average Precision on the hold-out test set)
- Interpretation: permutation feature importance (measures drop in a chosen score when a feature is shuffled)
- Robustness check for multicollinearity: `late_30_59`, `late_60_89`, `late_90` are extremely correlated, so I retrained a reduced model keeping only `late_90`; performance changed by ~0.03, suggesting the delinquency-bucket signals are largely shared
- Calibration: reliability curve + Brier score to evaluate whether predicted probabilities match observed outcome rates

## Key results (test set)
- Logistic Regression: ROC-AUC ≈ 0.79, Average Precision ≈ 0.81
- HistGradientBoosting (all features): ROC-AUC ≈ 0.86, Average Precision ≈ 0.86
- HistGradientBoosting (reduced: keep `late_90` only): performance within ~0.03 of the full model
- Brier score loss (reduced HGB): 0.1689 (lower is better probability accuracy)

## Feature insights
Permutation importance indicates the model relies most on `rev_util` and delinquency history features (especially `late_90`), while variables like `dependents` contribute little in this dataset. Importance here describes what helped the trained model generalize on the test set; it should not be interpreted as causality.

## Notes / limitations
- The dataset appears perfectly balanced (50/50), which is not typical in real credit-default settings; real-world thresholding and business costs would matter more than a default 0.5 cutoff
- High collinearity among delinquency-bucket features means importance is shared and should not be interpreted as independent effects

## How to run
1. Download the dataset CSV from Kaggle
2. Place it in `data/` (or adjust the path in the notebook)
3. Create an environment and install dependencies:
   - `pip install -r requirements.txt`
4. Run `data-exploration.ipynb` to reproduce training, evaluation (ROC/PR), permutation importance, correlation matrix, and calibration

## Repo structure (suggested)
- `data-exploration.ipynb` — end-to-end EDA + modeling + evaluation
- `models/` — saved model bundle (joblib) + feature list for deployment
- `requirements.txt` — dependencies
