# Pre-approval credit-risk tutorial

Open `Tutorial_revised.ipynb` after placing it in the same directory as `lending_club_tutorial.csv`. The notebook predicts default using information available before approval and pricing.

The model inputs are `person_age`, `annual_inc`, `home_ownership`, `emp_length`, `purpose`, `prior_default`, `credit_history`, `loan_amnt` (the requested amount), and a recomputed `loan_percent_income = loan_amnt / annual_inc`. `int_rate` and `grade` are excluded because they are pricing or lender-assigned risk outputs. `default` is the target only.

The notebook keeps the original LR and RF baseline parameters and uses the frozen XGBoost parameters selected by five-fold training cross-validation. The included test comparison gives AUC 0.8037 for LR, 0.8631 for RF, and 0.9036 for XGBoost at threshold 0.50.

## Contents

- `Tutorial_revised.ipynb`: renamed v6 teaching notebook, with outputs and high-resolution plotting cells.
- `lending_club_tutorial.csv`: the data used by the notebook.
- `figures/`: 600 dpi PNG and PDF confusion matrices, ROC curves, and threshold-cost plots.
- `tune_preapproval_credit_risk.py`: train-only XGBoost search and reproducibility script.
- `publish_preapproval_tutorial.py`: notebook generation and execution/validation script.
- `preapproval_results/`: selected parameters, feature audit, trials, metrics, and execution log.

## Reproduce

From this folder, use a Python environment containing the packages in `requirements_auc.txt`:

```powershell
python tune_preapproval_credit_risk.py --trials 48
python publish_preapproval_tutorial.py
```

The search uses a fixed seed and keeps identical allowed-input records in the same split. The final test set is not used to choose model parameters or thresholds.
