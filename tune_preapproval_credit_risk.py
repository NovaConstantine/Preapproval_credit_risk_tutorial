"""Pre-approval / pre-pricing tutorial: explicit feature allowlist and train-only search.

Run: .venv_auc/Scripts/python.exe tune_preapproval_credit_risk.py
Rerun the frozen configuration: add --refit (no new parameter selection).
The owner confirms loan_amnt is the REQUESTED amount. Historical default and
credit-history fields are assumed to be from the application-time credit file.
The supplied Imperial source describes the data as synthetic teaching data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
OUT = HERE / 'preapproval_results'
SEED = 42
NUM = ['person_age', 'annual_inc', 'emp_length', 'prior_default',
       'credit_history', 'loan_amnt', 'loan_percent_income']
CAT = ['home_ownership', 'purpose']
FEATURES = NUM + CAT
BASELINES = {
    'LR': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000},
    'RF': {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 20,
           'class_weight': 'balanced'},
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def make_features(df):
    """Never encode the entire CSV: keep only application-time inputs."""
    raw = [name for name in FEATURES if name != 'loan_percent_income']
    X = df.loc[:, raw].copy()
    income = pd.to_numeric(X['annual_inc'], errors='raise')
    amount = pd.to_numeric(X['loan_amnt'], errors='raise')
    # Invalid income produces a missing ratio, imputed inside the training fold.
    X['loan_percent_income'] = amount.div(income.where(income > 0))
    X = X[FEATURES].replace([np.inf, -np.inf], np.nan)
    assert not {'default', 'int_rate', 'grade'}.intersection(X.columns)
    return X


def split_data(X, y):
    # Identical allowed-input rows stay together, including at the outer split.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.30,
                                    random_state=SEED).split(X, y, groups))
    assert not set(groups[tr]).intersection(groups[te])
    return tr, te, groups


def preprocessor():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUM),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore',
                                      sparse_output=False)),
        ]), CAT),
    ], remainder='drop', sparse_threshold=0)


def estimator(name, params):
    if name == 'LR':
        return LogisticRegression(random_state=SEED, **params)
    if name == 'RF':
        return RandomForestClassifier(random_state=SEED, n_jobs=2, **params)
    return XGBClassifier(objective='binary:logistic', eval_metric='auc',
        tree_method='hist', random_state=SEED, n_jobs=2, verbosity=0, **params)


def model_pipeline(name, params):
    steps = [('preprocess', preprocessor())]
    if name == 'LR':
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps + [('model', estimator(name, params))])


def prepare_folds(X, y, groups):
    cache = []
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in folds.split(X, y, groups):
        assert not set(groups[tr]).intersection(groups[va])
        prep = preprocessor()
        a, b = prep.fit_transform(X.iloc[tr]), prep.transform(X.iloc[va])
        scale = StandardScaler().fit(a)
        cache.append((tr, va, a, b, scale.transform(a), scale.transform(b)))
    return cache


def oof_predict(name, params, cache, y):
    prob = np.full(len(y), np.nan)
    aucs = []
    for tr, va, a, b, sa, sb in cache:
        model = estimator(name, params)
        model.fit(sa if name == 'LR' else a, y[tr])
        prob[va] = model.predict_proba(sb if name == 'LR' else b)[:, 1]
        aucs.append(float(roc_auc_score(y[va], prob[va])))
    assert np.isfinite(prob).all()
    return prob, aucs


def metrics(y, prob, threshold):
    tn, fp, fn, tp = confusion_matrix(y, prob >= threshold, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, prob)), 'threshold': float(threshold),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'error_rate': float((fp + fn) / len(y)),
            'default_recall': float(tp / (tp + fn)),
            'default_precision': float(tp / (tp + fp))}


def search_params(t):
    return {
        'n_estimators': t.suggest_int('n_estimators', 300, 1200, step=100),
        'learning_rate': t.suggest_float('learning_rate', .025, .16, log=True),
        'max_depth': t.suggest_int('max_depth', 2, 7),
        'min_child_weight': t.suggest_float('min_child_weight', 1, 30, log=True),
        'subsample': t.suggest_float('subsample', .70, 1.0),
        'colsample_bytree': t.suggest_float('colsample_bytree', .70, 1.0),
        'reg_alpha': t.suggest_float('reg_alpha', .0001, 10, log=True),
        'reg_lambda': t.suggest_float('reg_lambda', .10, 30, log=True),
        'gamma': t.suggest_float('gamma', 0, 1),
        'scale_pos_weight': t.suggest_float('scale_pos_weight', 1, 5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=48)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / 'lending_club_tutorial.csv')
    X, y = make_features(df), df['default'].astype(int).to_numpy()
    tr, te, groups = split_data(X, y)
    Xtr, ytr = X.iloc[tr], y[tr]
    data_hash = hashlib.sha256((HERE / 'lending_club_tutorial.csv').read_bytes()).hexdigest()
    np.savez(OUT / 'split_indices.npz', train=tr, test=te)
    print(f'Inputs: {FEATURES}; train={len(tr)}, test={len(te)}', flush=True)
    cache = prepare_folds(Xtr, ytr, groups[tr])
    config_path = OUT / 'selected_params.json'
    oof = {}
    for name, params in BASELINES.items():
        oof[name], aucs = oof_predict(name, params, cache, ytr)
        print(f'{name} unchanged parameters, new inputs: CV AUC={np.mean(aucs):.6f}', flush=True)

    if args.refit:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        assert config['data_sha256'] == data_hash
        assert config['features'] == FEATURES
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=12))
        for depth, weight in [(3, 1.0), (5, 1.0), (5, 3.5), (4, 2.0)]:
            study.enqueue_trial({'n_estimators': 800, 'learning_rate': .05,
                'max_depth': depth, 'min_child_weight': 5., 'subsample': .9,
                'colsample_bytree': .9, 'reg_alpha': .1, 'reg_lambda': 5.,
                'gamma': 0., 'scale_pos_weight': weight})

        def objective(trial):
            params = search_params(trial)
            prob, aucs = oof_predict('XGBoost', params, cache, ytr)
            trial.set_user_attr('fold_auc', aucs)
            np.save(OUT / f'trial_{trial.number:03d}_oof.npy', prob)
            return float(np.mean(aucs))

        def checkpoint(study, trial):
            study.trials_dataframe().to_csv(OUT / 'xgb_trials.csv', index=False)
            print(f'XGBoost {trial.number+1}/{args.trials}: AUC={trial.value:.6f}; '
                  f'best={study.best_value:.6f}', flush=True)

        study.optimize(objective, n_trials=args.trials, callbacks=[checkpoint])
        best = study.best_trial
        oof['XGBoost'] = np.load(OUT / f'trial_{best.number:03d}_oof.npy')
        # Freeze the model by mean CV AUC, then choose an operating threshold
        # from TRAINING OOF scores. Test labels are never used for selection.
        rf = metrics(ytr, oof['RF'], .5)
        candidates = []
        for threshold in np.linspace(.05, .95, 181):
            row = metrics(ytr, oof['XGBoost'], threshold)
            row['worst_error_ratio_to_rf'] = max(row['fp']/rf['fp'], row['fn']/rf['fn'])
            candidates.append(row)
        threshold_table = pd.DataFrame(candidates).sort_values(
            ['worst_error_ratio_to_rf', 'error_rate', 'threshold'])
        threshold_table.to_csv(OUT / 'training_threshold_search.csv', index=False)
        chosen = float(threshold_table.iloc[0]['threshold'])
        config = {'data_sha256': data_hash, 'seed': SEED, 'features': FEATURES,
            'excluded': ['int_rate', 'grade'],
            'ratio_definition': 'requested loan_amnt / annual_inc; recomputed',
            'outer_split': 'GroupShuffleSplit(test_size=0.30, random_state=42); identical X rows grouped',
            'cv': '5-fold StratifiedGroupKFold on training data; preprocessing fitted within each fold',
            'selection': 'maximum mean CV AUC; then min max(FP/RF_FP, FN/RF_FN) on training OOF',
            'models': {**BASELINES, 'XGBoost': best.params},
            'xgb_cv_auc_mean': best.value, 'xgb_fold_auc': best.user_attrs['fold_auc'],
            'xgb_trial_number': best.number, 'trials': args.trials,
            'thresholds': {'LR': .5, 'RF': .5, 'XGBoost': chosen},
            'sample_sizes': {'train': len(tr), 'test': len(te)},
            'versions': {n: importlib.metadata.version(n) for n in
                         ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'optuna']}}
        # This file is written before evaluating any model on the test set.
        save_json(config_path, config)

    if 'XGBoost' not in oof:
        oof['XGBoost'], _ = oof_predict('XGBoost', config['models']['XGBoost'], cache, ytr)
    pd.DataFrame({'row_index': tr, 'default': ytr, **oof}).to_csv(
        OUT / 'training_oof_predictions.csv', index=False)
    rows, predictions = [], {'row_index': te, 'default': y[te]}
    cv_rows = []
    for name, params in config['models'].items():
        model = model_pipeline(name, params).fit(Xtr, ytr)
        names = model.named_steps['preprocess'].get_feature_names_out().tolist()
        assert not any('grade' in n or 'int_rate' in n for n in names)
        save_json(OUT / 'encoded_feature_names.json', names)
        prob = model.predict_proba(X.iloc[te])[:, 1]
        predictions[name] = prob
        rows.append({'model': name, 'operating_point': '0.50', **metrics(y[te], prob, .5)})
        cv_rows.append({'model': name, **metrics(ytr, oof[name], config['thresholds'][name])})
        if name == 'XGBoost':
            rows.append({'model': name, 'operating_point': 'training-selected',
                         **metrics(y[te], prob, config['thresholds'][name])})
    pd.DataFrame(rows).to_csv(OUT / 'test_metrics.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / 'training_oof_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'test_predictions.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
