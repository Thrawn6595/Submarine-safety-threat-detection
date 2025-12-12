"""
Model Training and Evaluation Utilities
Cost-sensitive classification for submarine mine detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, precision_score, make_scorer, confusion_matrix, accuracy_score


def calculate_cost(y_true: np.ndarray, y_pred: np.ndarray, 
                  fn_cost: int = 100, fp_cost: int = 1) -> float:
    """Calculate total cost with asymmetric penalties"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    return total_cost


def cost_scorer(fn_cost: int = 100, fp_cost: int = 1):
    """Custom scorer for GridSearchCV - returns negative cost for maximization"""
    def scorer(y_true, y_pred):
        cost = calculate_cost(y_true, y_pred, fn_cost, fp_cost)
        return -cost
    return make_scorer(scorer)


def get_model_configurations() -> Dict:
    """Define models with hyperparameter grids for tuning"""
    configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 20}, {0: 1, 1: 50}]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 20}]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 20}, 'balanced']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'param_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },
        'LDA': {
            'model': LinearDiscriminantAnalysis(),
            'param_grid': {
                'solver': ['svd', 'lsqr']
            }
        },
        'QDA': {
            'model': QuadraticDiscriminantAnalysis(),
            'param_grid': {
                'reg_param': [0.0, 0.1, 0.3, 0.5]
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(random_state=42, verbose=False),
            'param_grid': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'depth': [4, 6, 8],
                'class_weights': [[1, 10], [1, 20]]
            }
        }
    }
    return configs


def tune_and_evaluate_model(model_name: str, model, param_grid: Dict,
                            X_train: np.ndarray, y_train: np.ndarray,
                            cv: int = 5, fn_cost: int = 100, fp_cost: int = 1) -> Dict:
    """Hyperparameter tuning with GridSearchCV and cross-validation evaluation"""
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring=cost_scorer(fn_cost, fp_cost),
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    recall_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='recall')
    precision_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='precision')
    
    y_pred_cv = []
    y_true_cv = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        best_model.fit(X_fold_train, y_fold_train)
        y_pred_cv.extend(best_model.predict(X_fold_val))
        y_true_cv.extend(y_fold_val)
    
    cv_cost = calculate_cost(np.array(y_true_cv), np.array(y_pred_cv), fn_cost, fp_cost)
    
    results = {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'cv_recall_mean': recall_scores.mean(),
        'cv_recall_std': recall_scores.std(),
        'cv_precision_mean': precision_scores.mean(),
        'cv_precision_std': precision_scores.std(),
        'cv_cost': cv_cost
    }
    
    return results


def evaluate_final_model(model, X_test: np.ndarray, y_test: np.ndarray,
                        model_name: str, fn_cost: int = 100, fp_cost: int = 1) -> Dict:
    """Final evaluation on test set"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    total_cost = calculate_cost(y_test, y_pred, fn_cost, fp_cost)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'total_cost': total_cost,
        'fn': fn,
        'fp': fp
    }
    
    return results


def create_cv_results_table(cv_results_list: List[Dict]) -> pd.DataFrame:
    """Create styled CV results comparison table"""
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'CV_Recall': f"{r['cv_recall_mean']:.3f} ± {r['cv_recall_std']:.3f}",
        'CV_Precision': f"{r['cv_precision_mean']:.3f} ± {r['cv_precision_std']:.3f}",
        'CV_Cost': r['cv_cost']
    } for r in cv_results_list])
    
    df = df.sort_values('CV_Cost', ascending=True)
    
    styled = df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '11pt'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), 
                                     ('background-color', '#4472C4'), 
                                     ('color', 'white'),
                                     ('text-align', 'center')]}
    ]).hide(axis='index')
    
    return styled


def create_test_results_table(test_results_list: List[Dict]) -> pd.DataFrame:
    """Create styled test results table"""
    df = pd.DataFrame(test_results_list)
    df = df[['model', 'accuracy', 'recall', 'precision', 'fn', 'fp', 'total_cost']]
    df = df.sort_values('total_cost', ascending=True)
    
    def highlight_best(row):
        colors = []
        for col in df.columns:
            if col in ['accuracy', 'recall', 'precision']:
                if row[col] == df[col].max():
                    colors.append('background-color: #c6efce')
                else:
                    colors.append('')
            elif col in ['fn', 'fp', 'total_cost']:
                if row[col] == df[col].min():
                    colors.append('background-color: #c6efce')
                else:
                    colors.append('')
            else:
                colors.append('')
        return colors
    
    styled = df.style.apply(highlight_best, axis=1).format({
        'accuracy': '{:.3f}',
        'recall': '{:.3f}',
        'precision': '{:.3f}',
        'total_cost': '{:.0f}'
    }).set_properties(**{
        'text-align': 'center',
        'font-size': '11pt'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), 
                                     ('background-color', '#70AD47'), 
                                     ('color', 'white'),
                                     ('text-align', 'center')]}
    ]).hide(axis='index')
    
    return styled


def plot_cv_comparison(cv_results_list: List[Dict]):
    """Plot cross-validation comparison"""
    df = pd.DataFrame(cv_results_list)
    df = df.sort_values('cv_cost')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    colors_recall = ['darkgreen' if r >= 0.95 else 'orange' for r in df['cv_recall_mean']]
    ax1.barh(df['model_name'], df['cv_recall_mean'], xerr=df['cv_recall_std'],
             color=colors_recall, edgecolor='black', capsize=5)
    ax1.axvline(0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    ax1.set_xlabel('Recall (CV Mean ± Std)', fontweight='bold')
    ax1.set_title('Cross-Validation Recall', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(df['model_name'], df['cv_precision_mean'], xerr=df['cv_precision_std'],
             color='steelblue', edgecolor='black', capsize=5)
    ax2.set_xlabel('Precision (CV Mean ± Std)', fontweight='bold')
    ax2.set_title('Cross-Validation Precision', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    colors_cost = ['darkgreen' if c == df['cv_cost'].min() else 'orange' for c in df['cv_cost']]
    ax3.barh(df['model_name'], df['cv_cost'], color=colors_cost, edgecolor='black')
    ax3.set_xlabel('Total Cost (Lower = Better)', fontweight='bold')
    ax3.set_title('Cross-Validation Cost', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Cross-Validation Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_test_comparison(test_results_list: List[Dict]):
    """Plot final test set comparison"""
    df = pd.DataFrame(test_results_list)
    df = df.sort_values('total_cost')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    colors_recall = ['darkgreen' if r >= 0.95 else 'orange' for r in df['recall']]
    ax1.barh(df['model'], df['recall'], color=colors_recall, edgecolor='black')
    ax1.axvline(0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    ax1.set_xlabel('Recall', fontweight='bold')
    ax1.set_title('Test Set Recall', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(df['model'], df['precision'], color='steelblue', edgecolor='black')
    ax2.set_xlabel('Precision', fontweight='bold')
    ax2.set_title('Test Set Precision', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    colors_fn = ['green' if fn == 0 else 'red' for fn in df['fn']]
    ax3.barh(df['model'], df['fn'], color=colors_fn, edgecolor='black')
    ax3.set_xlabel('Missed Mines (False Negatives)', fontweight='bold')
    ax3.set_title('Safety Risk: Missed Mines', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Final Test Set Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
