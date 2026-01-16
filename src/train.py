import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
import joblib
import os

pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
study_balanced = optuna.create_study(direction='maximize', study_name='balanced_weight')
study_custom = optuna.create_study(direction='maximize', study_name='custom_weight')
study_none = optuna.create_study(direction='maximize', study_name='no_weight')

def load_split_data(filename='../data/creditcard.csv'):
    df = pd.read_csv(filename)
    
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        X_train_fold, X_val_fold = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_train_fold, y_val_fold = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
        print(y_train_fold[y_train_fold == 1].shape[0]/y_train_fold.shape[0])
        print(y_val_fold[y_val_fold == 1].shape[0]/y_val_fold.shape[0])

    return skf, X_trainval, X_test, y_trainval, y_test

def et_balanced_hp_tuning(skf, X_trainval, y_trainval, n_trials=200, show_progress_bar=True, save_pics=True,
                          opt_history_filename="../results/optuna/balanced/optimization_history_balanced.png",
                          param_importances_filename="../results/optuna/balanced/param_importances_custom_balanced.png",
                          model_filename="../models/et/model_et_balanced.pkl"
):
    def objective_balanced(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', low=50, high=300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=1, high=20),
            'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'class_weight': 'balanced',
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'max_samples': trial.suggest_float('max_samples', low=0.5, high=1.0) if trial.params['bootstrap'] else None,
            'random_state': 42,
            'n_jobs': -1,
            'criterion': 'gini',
        }
        
        params = {k: v for k, v in params.items() if v is not None}
        scores = []
        
        for train_idx, val_idx in skf.split(X_trainval, y_trainval):
            X_train_fold, X_val_fold = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
            y_train_fold, y_val_fold = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict_proba(X_val_fold)[:, 1]
            score = average_precision_score(y_val_fold, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study_balanced.optimize(objective_balanced, n_trials=n_trials, show_progress_bar=show_progress_bar,)

    if save_pics:
        fig_opt_history = plot_optimization_history(study_balanced)
        fig_opt_history.write_image(opt_history_filename, scale=2, width=1200, height=800)
        fig_param_importances = plot_param_importances(study_balanced)
        fig_param_importances.write_image(param_importances_filename, scale=2, width=1200, height=800)
    
    best_params_balanced = study_balanced.best_params
    complete_params = {
        **best_params_balanced,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'criterion': 'gini',
        'verbose': 0,
        'max_samples': None,
    }
    model_balanced = ExtraTreesClassifier(**complete_params)
    model_balanced.fit(X_trainval, y_trainval)
    joblib.dump(model_balanced, model_filename)



def et_custom_hp_tuning(skf, X_trainval, y_trainval, n_trials=200, show_progress_bar=True, save_pics=True,
                          opt_history_filename="../results/optuna/custom/optimization_history_custom.png",
                          param_importances_filename="../results/optuna/custom/param_importances_custom.png",
                          model_filename="../models/et/model_et_custom.pkl"
):
    def objective_custom_weight(trial):
        fraud_weight = trial.suggest_float('fraud_weight', low=10, high=500, log=True)
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', low=50, high=300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=1, high=20),
            'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'class_weight': {0: 1, 1: fraud_weight},
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'max_samples': trial.suggest_float('max_samples', low=0.5, high=1.0) if trial.params['bootstrap'] else None,
            'random_state': 42,
            'n_jobs': -1,
            'criterion': 'gini',
        }
        
        params = {k: v for k, v in params.items() if v is not None}
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_trainval, y_trainval):
            X_train_fold, X_val_fold = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
            y_train_fold, y_val_fold = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict_proba(X_val_fold)[:, 1]
            score = average_precision_score(y_val_fold, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study_custom.optimize(objective_custom_weight, n_trials=n_trials, show_progress_bar=show_progress_bar)

    if save_pics:
        fig_opt_history = plot_optimization_history(study_custom)
        fig_opt_history.write_image(opt_history_filename, scale=2, width=1200, height=800)
        fig_param_importances = plot_param_importances(study_custom)
        fig_param_importances.write_image(param_importances_filename, scale=2, width=1200, height=800)
    
    best_params_custom = study_custom.best_params
    fraud_weight = best_params_custom['fraud_weight']
    class_weight_dict = {0: 1, 1: fraud_weight}
    complete_params_custom = {
        'n_estimators': best_params_custom['n_estimators'],
        'max_depth': best_params_custom['max_depth'],
        'min_samples_leaf': best_params_custom['min_samples_leaf'],
        'min_samples_split': best_params_custom['min_samples_split'],
        'max_features': best_params_custom['max_features'],
        'bootstrap': best_params_custom['bootstrap'],
        'class_weight': class_weight_dict,
        'random_state': 42,
        'n_jobs': -1,
        'criterion': 'gini',
        'verbose': 0,
    }
    model_custom = ExtraTreesClassifier(**complete_params_custom)
    model_custom.fit(X_trainval, y_trainval)
    joblib.dump(model_custom, model_filename)



def et_none_hp_tuning(skf, X_trainval, y_trainval, n_trials=200, show_progress_bar=True, save_pics=True,
                          opt_history_filename="../results/optuna/none/optimization_history_none.png",
                          param_importances_filename="../results/optuna/none/param_importances_none.png",
                          model_filename="../models/et/model_et_none.pkl"
):
    def objective_no_weight(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', low=50, high=300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 20, 30, 40, 50, 60]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=1, high=20),
            'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
            'class_weight': None,
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'max_samples': trial.suggest_float('max_samples', low=0.5, high=1.0) if trial.params['bootstrap'] else None,
            'random_state': 42,
            'n_jobs': -1,
            'criterion': 'gini',
        }
        
        params = {k: v for k, v in params.items() if v is not None}
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_trainval, y_trainval):
            X_train_fold, X_val_fold = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
            y_train_fold, y_val_fold = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict_proba(X_val_fold)[:, 1]
            score = average_precision_score(y_val_fold, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study_none.optimize(objective_no_weight, n_trials=n_trials, show_progress_bar=show_progress_bar)

    if save_pics:
        fig_opt_history = plot_optimization_history(study_none)
        fig_opt_history.write_image(opt_history_filename, scale=2, width=1200, height=800)
        fig_param_importances = plot_param_importances(study_none)
        fig_param_importances.write_image(param_importances_filename, scale=2, width=1200, height=800)
    
    best_params_none = study_none.best_params
    complete_params_none = {
        'n_estimators': best_params_none['n_estimators'],
        'max_depth': best_params_none['max_depth'],
        'min_samples_leaf': best_params_none['min_samples_leaf'],
        'min_samples_split': best_params_none['min_samples_split'],
        'max_features': best_params_none['max_features'],
        'bootstrap': best_params_none['bootstrap'],
        'class_weight': None,
        'random_state': 42,
        'n_jobs': -1,
        'criterion': 'gini',
        'verbose': 0,
    }
    model_none = ExtraTreesClassifier(**complete_params_none)
    model_none.fit(X_trainval, y_trainval)
    joblib.dump(model_none, model_filename)
    
def __main__():
    skf, X_trainval, X_test, y_trainval, y_test = load_split_data()
