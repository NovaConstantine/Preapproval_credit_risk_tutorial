"""Build and execute the self-contained v6 notebook using frozen training selections."""
from __future__ import annotations
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig_auc')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = HERE / 'preapproval_results'
config = json.loads((OUT / 'selected_params.json').read_text(encoding='utf-8'))
notebook = json.loads((HERE / 'Tutorial_revised_v5_template.ipynb').read_text(encoding='utf-8'))


def set_cell(i, text):
    cell = notebook['cells'][i]
    cell['source'] = textwrap.dedent(text).strip().splitlines(keepends=True)
    # Ensure the last source line has a newline for normal Jupyter editing.
    cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'


set_cell(0, '''
# Tutorial 4: Classification for Credit Risk — v6

## Risk prediction before approval and pricing

Build Logistic Regression, Random Forest and XGBoost models using information
available at application. Compare ranking (AUC) and classification errors, then
choose a screening threshold using training data only.

This tutorial adapts the earlier classification exercises. The supplied Imperial
source notebook describes the underlying credit dataset as **synthetic teaching
data**. The filename `lending_club_tutorial.csv` does not establish that it is real
LendingClub data. This example does not reproduce Bian's separate raw-data analysis.
''')
set_cell(1, '''
**Scope:** Predict subsequent default using an application-time information set.
This dataset contains outcomes for recorded loans, not outcomes for every rejected
applicant. Its results illustrate classification and do not establish performance
for a live applicant population. No outcome horizon or application-time snapshot
is supplied; the timing assumptions below are part of the teaching setup.
''')
set_cell(2, '''
## Part 1: Setup and Data Preparation

**Feature timing rules**

- Exclude `int_rate` and `grade`: final pricing and lender-assigned risk assessments
  are unavailable before approval and pricing.
- Retain `loan_amnt` as the **requested amount**, as confirmed by the dataset owner.
- Recompute `loan_percent_income = loan_amnt / annual_inc` from application inputs;
  it is a requested-loan-to-annual-income ratio, **not DTI**.
- Assume age, income, housing, employment, purpose, prior default and credit-history
  length refer to the application date. `prior_default` is historical credit-file
  information, not the outcome of this loan, and is not a FICO score.
- Select inputs by an explicit allowlist. Fit imputation, encoding and scaling on
  training data only. Keep identical allowed-input records in the same partition.

LR and RF keep their earlier model parameters. XGBoost parameters are selected by
five-fold training CV AUC. A separate XGBoost screening threshold is selected from
training out-of-fold scores to balance FP and FN relative to RF. Neither selection
uses the test labels. CV selection scores are optimistic; the test split is an
internal teaching check, not new external or temporal validation.
''')
set_cell(3, '''
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
    cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)
print('Libraries loaded. Prediction time: before approval and pricing.')
''')
set_cell(4, '''
# The filename is retained for compatibility; the source describes synthetic data.
data_path = Path('lending_club_tutorial.csv')
df = pd.read_csv(data_path)
print(f'Dataset shape: {df.shape}')
print(df.columns.tolist())
''')
set_cell(7, '''
# Eight recorded application inputs plus one derived application-time ratio.
numeric_cols = ['person_age', 'annual_inc', 'emp_length', 'prior_default',
                'credit_history', 'loan_amnt', 'loan_percent_income']
categorical_cols = ['home_ownership', 'purpose']
input_cols = numeric_cols + categorical_cols
raw_cols = [name for name in input_cols if name != 'loan_percent_income']
X = df.loc[:, raw_cols].copy()
income = pd.to_numeric(X['annual_inc'], errors='raise')
X['loan_percent_income'] = X['loan_amnt'].div(income.where(income > 0))
X = X[input_cols].replace([np.inf, -np.inf], np.nan)
y = df['default'].astype(int)
assert not {'default', 'int_rate', 'grade'}.intersection(X.columns)
assert X.columns.tolist() == input_cols
print('Raw model inputs:', X.columns.tolist())
print('Excluded: int_rate, grade. Target only: default.')
X.describe().round(3)
''')
set_cell(8, '''
# Define preprocessing now; fit it only after splitting.
# This function also creates a fresh transformer for every CV fold.
def make_preprocessor():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric_cols),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore',
                                      sparse_output=False)),
        ]), categorical_cols),
    ], remainder='drop', sparse_threshold=0)
''')
set_cell(9, '''
print('Missing values in allowed inputs:')
print(X.isna().sum())
# Missing values are filled using training-fold statistics, never test statistics.
''')
set_cell(10, '''
# Group identical application records to prevent overlap across partitions.
# These groups do not use the target, interest rate or grade.
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
train_idx, test_idx = next(GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42).split(X, y, groups))
assert not set(groups[train_idx]).intersection(groups[test_idx])
X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_groups = groups[train_idx]
print(f'Train: {len(train_idx):,}; test: {len(test_idx):,}')
print(f'Default rates: train={y_train.mean():.2%}; test={y_test.mean():.2%}')
''')
set_cell(11, '''
preprocess = make_preprocessor()
X_train = preprocess.fit_transform(X_train_raw)
X_test = preprocess.transform(X_test_raw)
feature_names = preprocess.get_feature_names_out().tolist()
assert not any('grade' in name or 'int_rate' in name for name in feature_names)
assert X_train.shape[1] == X_test.shape[1] == len(feature_names)
print(f'Encoded features ({len(feature_names)}):')
print(feature_names)
''')
set_cell(12, '''
# Scale the encoded features for LR only, using the training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
''')
set_cell(13, '''
## Part 2: Train Classification Models

1. Logistic Regression: original baseline parameters, retrained on the new inputs.
2. Random Forest: original baseline parameters, retrained on the same inputs.
3. XGBoost: parameters frozen after a search using the training set only.

All models see the same information. Weighted model scores are useful for ranking
and thresholding but should not be interpreted directly as calibrated PDs.
''')
set_cell(14, '''
# Same baseline: default L2 regularization, C=1, balanced weights, seed=42.
logreg = LogisticRegression(C=1.0, class_weight='balanced',
                            max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
print('Logistic Regression fitted.')
''')
set_cell(15, '''
# Same RF baseline; n_jobs only controls CPU use.
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
    min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=2)
rf.fit(X_train, y_train)
print('Random Forest fitted.')
''')
set_cell(16, f'''
# Frozen by training CV AUC; do not retune using the test results below.
xgb_params = {repr(config['models']['XGBoost'])}
xgb_threshold = {repr(config['thresholds']['XGBoost'])}
expected_data_sha256 = {repr(config['data_sha256'])}
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == expected_data_sha256, \\
    'Dataset changed: rerun tune_preapproval_credit_risk.py before using these results.'
xgb = XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='auc',
    tree_method='hist', random_state=42, n_jobs=2, verbosity=0)
xgb.fit(X_train, y_train)
print('XGBoost fitted. Training-selected screening threshold:', xgb_threshold)
''')
set_cell(17, '''
y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_logreg = (y_prob_logreg >= 0.50).astype(int)
y_pred_rf = (y_prob_rf >= 0.50).astype(int)
y_pred_xgb = (y_prob_xgb >= 0.50).astype(int)
y_pred_xgb_screening = (y_prob_xgb >= xgb_threshold).astype(int)
print('Predictions generated. Main comparison uses 0.50 for every model.')
''')
set_cell(18, '''
## Part 3: The Confusion Matrix

Rows are actual outcomes; columns are predicted outcomes. Top right is a false
positive (a non-defaulter flagged as risky); bottom left is a false negative
(a missed default). The first figure uses 0.50 for all models. The second displays
the frozen training-selected XGBoost screening threshold; LR and RF stay at 0.50.
Threshold changes do not change AUC.
''')
plot_source = ''.join(notebook['cells'][19]['source'])
plot_source = plot_source[plot_source.index('model_predictions ='):]
plot_source = plot_source.replace("('XGBoost', y_pred_xgb),", "(xgb_label, xgb_prediction),")
plot_source = plot_source.replace("fig.suptitle('Confusion Matrices at threshold = 0.50', fontsize=18, y=0.95)",
                                 "fig.suptitle(figure_title, fontsize=18, y=0.95)")
plot_source = plot_source.replace("'confusion_matrices_v5_600dpi.png'", "f'{stem}_600dpi.png'")
plot_source = plot_source.replace("'confusion_matrices_v5.pdf'", "f'{stem}.pdf'")
set_cell(19, '''
# Fixed fourth column for one shared colorbar; white separators at category edges.
def plot_confusion_comparison(xgb_prediction, xgb_label, figure_title, stem):
''' + textwrap.indent(plot_source, '    ') + '''

plot_confusion_comparison(y_pred_xgb, 'XGBoost',
    'Pre-approval risk: all thresholds = 0.50', 'confusion_matrices_v6_050')
plot_confusion_comparison(y_pred_xgb_screening, f'XGBoost (t = {xgb_threshold:.3f})',
    'Pre-approval risk: training-selected XGBoost threshold; LR / RF = 0.50',
    'confusion_matrices_v6_screening')
''')
set_cell(20, '''
TN, FP, FN, TP = confusion_matrix(y_test, y_pred_logreg, labels=[0, 1]).ravel()
print(f'LR: TN={TN:,}, FP={FP:,}, FN={FN:,}, TP={TP:,}')
print('FP: non-defaulter flagged as risky. FN: missed default.')
''')
roc_source = ''.join(notebook['cells'][25]['source'])
roc_source = roc_source.replace("plt.title('ROC Curves: Model Comparison')", "plt.title('Pre-approval Risk: ROC Curves')")
roc_source = roc_source.replace('plt.show()', "plt.savefig(output_dir / 'roc_curves_v6_600dpi.png', dpi=600, facecolor='white')\nplt.savefig(output_dir / 'roc_curves_v6.pdf', facecolor='white')\nplt.show()")
set_cell(25, roc_source)
set_cell(28, '''
## Part 5: Threshold Selection Using Training Data

AUC evaluates ranking. An operating threshold trades off false alarms and missed
defaults. The 0.50 threshold is a reference point, not necessarily the best business
choice. Here we illustrate a **different objective** from the balanced-error figure:
minimize an assumed cost of FP and FN, using training out-of-fold predictions only.
The cost values are teaching assumptions, not measured losses or quoted loan terms.
''')
set_cell(29, '''
COST_FP = 500   # Illustrative cost units for a non-defaulter flagged as risky.
COST_FN = 5000  # Illustrative cost units for a missed default.
print(f'Illustrative costs: FP={COST_FP}, FN={COST_FN}; ratio={COST_FN/COST_FP:.0f}')
''')
cost_function = ''.join(notebook['cells'][30]['source']).replace(
    'confusion_matrix(y_true, y_pred)', 'confusion_matrix(y_true, y_pred, labels=[0, 1])')
set_cell(30, cost_function)
set_cell(31, '''
# Each training prediction comes from a model that did not fit that row or group.
# The preprocessing is also fitted separately inside each fold.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv.split(X_train_raw, y_train, train_groups))
cv_model = Pipeline([
    ('preprocess', make_preprocessor()),
    ('model', XGBClassifier(**xgb_params, objective='binary:logistic',
        eval_metric='auc', tree_method='hist', random_state=42, n_jobs=2, verbosity=0)),
])
y_prob_train_oof = cross_val_predict(cv_model, X_train_raw, y_train,
    cv=cv_splits, method='predict_proba', n_jobs=1)[:, 1]
results_df = pd.DataFrame([
    calculate_cost_at_threshold(y_train, y_prob_train_oof, t, COST_FP, COST_FN)
    for t in np.linspace(0.05, 0.95, 181)
])
optimal_idx = results_df['total_cost'].idxmin()
optimal_threshold = float(results_df.loc[optimal_idx, 'threshold'])
optimal_cost = float(results_df.loc[optimal_idx, 'total_cost'])
print(f'Training-selected cost threshold: {optimal_threshold:.3f}')
print('Selection used training OOF scores only; these selection scores are optimistic.')
''')
cost_plot = ''.join(notebook['cells'][32]['source'])
cost_plot = cost_plot.replace('Total Cost (£ millions)', 'Illustrative cost (million units)')
cost_plot = cost_plot.replace('Total Cost vs Classification Threshold', 'Training OOF: Cost vs Threshold')
cost_plot = cost_plot.replace('Error Counts vs Threshold', 'Training OOF: Error Counts')
cost_plot = cost_plot.replace('Optimal (', 'CV-selected (')
cost_plot = cost_plot.replace('plt.show()', "plt.savefig(output_dir / 'threshold_cost_v6_600dpi.png', dpi=600, facecolor='white')\nplt.savefig(output_dir / 'threshold_cost_v6.pdf', facecolor='white')\nplt.show()")
set_cell(32, cost_plot)
set_cell(33, '''
# Evaluate frozen thresholds; do not choose between them using these test results.
comparison = pd.DataFrame([
    {'Policy': label, **calculate_cost_at_threshold(
        y_test, y_prob_xgb, threshold, COST_FP, COST_FN)}
    for label, threshold in [('Reference', 0.50),
                              ('Balanced errors (training)', xgb_threshold),
                              ('Cost objective (training)', optimal_threshold)]
])
print('Test results at thresholds selected before test evaluation:')
print(comparison.to_string(index=False))
''')
set_cell(34, '''
### AI Prompt Exercise 2: Explain Threshold Selection

Explain how changing a risk-score threshold affects FP and FN. Use the displayed
reference and training-selected results. Why must threshold selection use training
or validation data, rather than the final test set? Why can a cost-minimizing
threshold differ from one that balances FP and FN relative to another model?
''')
set_cell(36, '''
## Part 6: From Risk Scores to Application Screening

This is a screening illustration before final approval or pricing. High scores
trigger further review. The model does not directly set interest rates or estimate
portfolio losses: that requires calibrated PDs, an outcome horizon and additional
assumptions about exposure, recovery, costs and loan terms.
''')
set_cell(37, '''
def screening_action(risk_score, threshold):
    return 'Further risk review' if risk_score >= threshold else 'Standard review'
''')
set_cell(38, '''
decisions_df = pd.DataFrame({
    'actual_default': y_test.to_numpy(),
    'risk_score': y_prob_xgb,
    'action': [screening_action(p, xgb_threshold) for p in y_prob_xgb],
})
decision_summary = decisions_df.groupby('action').agg(
    Count=('actual_default', 'size'),
    Defaults=('actual_default', 'sum'),
    Observed_default_rate=('actual_default', 'mean'),
)
print(decision_summary.to_string())
''')
set_cell(39, '''
# Requested amount is available, but it is not an approved exposure or a loss.
decisions_df['requested_amount'] = X_test_raw['loan_amnt'].to_numpy()
print(decisions_df.groupby('action')['requested_amount'].agg(['count', 'sum', 'mean']))
print('Class-weighted risk scores are not automatically calibrated probabilities of default.')
''')
set_cell(40, '''
### AI Prompt Exercise 3: Check the Information Set

Explain why `int_rate` and `grade` are excluded from a pre-approval, pre-pricing
model. Why can the requested amount be retained? Explain why the requested-loan-to-
income ratio is not DTI, and why historical default is not a FICO score. What extra
data and validation would be needed to use these scores for actual pricing?
''')
set_cell(43, '''
rows = []
for name, scores, threshold in [
    ('Logistic Regression', y_prob_logreg, 0.50),
    ('Random Forest', y_prob_rf, 0.50),
    ('XGBoost (0.50)', y_prob_xgb, 0.50),
    ('XGBoost (training-selected)', y_prob_xgb, xgb_threshold),
]:
    tn, fp, fn, tp = confusion_matrix(y_test, scores >= threshold, labels=[0, 1]).ravel()
    rows.append({'Model': name, 'AUC': roc_auc_score(y_test, scores),
        'Threshold': threshold, 'FP': fp, 'FN': fn,
        'Error rate': (fp + fn) / len(y_test)})
final_comparison = pd.DataFrame(rows)
print(final_comparison.to_string(index=False))
''')
set_cell(44, '''
### Key Lessons

1. Define prediction time before choosing features. Pricing and lender-assigned
   grade cannot be inputs to a model operating before those decisions.
2. All models use the same allowed information. LR and RF remain fixed baselines;
   XGBoost receives a training-only parameter search. This is not a comparison of
   equally tuned algorithms.
3. AUC measures ranking; FP and FN also depend on the operating threshold.
4. Choose parameters and thresholds inside the training data, then evaluate the
   frozen choices. Neither better AUC nor lower errors on both sides is guaranteed.
5. Weighted scores need calibration checks before interpretation as PDs. This
   synthetic loan sample does not establish live applicant or pricing performance.
''')

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
notebook['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
dest = HERE / 'Tutorial_revised_v6.ipynb'
dest.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')

# Execute all code cells, capture outputs and preview images, then save the notebook.
scope = {'__name__': '__main__'}
log = []
count = 0
with threadpool_limits(limits=2):
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        text_output = io.StringIO()
        pictures = []

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format='png', dpi=120, facecolor='white')
                pictures.append({'output_type': 'display_data', 'metadata': {},
                    'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}})
                plt.close(figure)

        plt.show = show
        with contextlib.redirect_stdout(text_output), contextlib.redirect_stderr(text_output):
            tree = ast.parse(''.join(cell['source']))
            last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, f'v6_cell_{i}', 'exec'), scope)
            if last_expr:
                value = eval(compile(ast.Expression(last_expr.value), f'v6_cell_{i}', 'eval'), scope)
                if value is not None:
                    print(value)
        cell['execution_count'] = count
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': text_output.getvalue()}]
                           if text_output.getvalue() else []) + pictures
        log.append(f'CELL {i}\n{text_output.getvalue()}')
        print(f'Executed cell {i}', flush=True)

# Check notebook and tuning script produce identical fitted probabilities and OOF.
expected = pd.read_csv(OUT / 'test_predictions.csv')
np.testing.assert_array_equal(expected.row_index, scope['test_idx'])
for name, key in [('LR', 'y_prob_logreg'), ('RF', 'y_prob_rf'), ('XGBoost', 'y_prob_xgb')]:
    np.testing.assert_allclose(expected[name], scope[key], rtol=1e-6, atol=1e-7)
expected_oof = pd.read_csv(OUT / 'training_oof_predictions.csv')
np.testing.assert_allclose(expected_oof.XGBoost, scope['y_prob_train_oof'], rtol=1e-6, atol=1e-7)
notebook['metadata']['preapproval_validation'] = {
    'all_cells_executed': True, 'predictions_match_search_refit': True,
    'data_sha256': config['data_sha256'], 'features': config['features']}
dest.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
(OUT / 'notebook_execution.log').write_text('\n\n'.join(log), encoding='utf-8')
scope['comparison'].to_csv(OUT / 'cost_threshold_test_metrics.csv', index=False)
print(f'Validated and saved {dest}', flush=True)
