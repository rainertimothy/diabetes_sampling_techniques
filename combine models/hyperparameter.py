from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def logistic(X_train, y_train):
    parameter = {
        "penalty": ["l1", "l2"],    
        "solver": ["saga"],              
        "max_iter": [50, 100, 200],                  
        "C": [1, 3, 5, 10],           
        "class_weight": [None, "balanced"]
    }

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logistic = LogisticRegression(random_state=42)

    tuning = RandomizedSearchCV(estimator=logistic, param_distributions=parameter, cv=fold, scoring="balanced_accuracy", verbose=2, n_jobs=-1, error_score='raise')

    model = tuning.fit(X_train, y_train)

    logistic_best_model = model.best_estimator_
    return logistic_best_model


def svm(X_train, y_train):
    parameter = {
        "C": [1, 3, 5, 10],                    
        "kernel": ["linear", "rbf", "poly"],             
        "class_weight": [None, "balanced"],         
        "tol": [0.001, 0.01],                              
    }

    svc = SVC(probability=True, random_state=42)

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm_tuning = RandomizedSearchCV(estimator=svc, param_distributions=parameter, cv=fold, scoring="balanced_accuracy", verbose=2, n_jobs=-1)

    model = svm_tuning.fit(X_train, y_train)

    svm_best_model = model.best_estimator_
    return svm_best_model

def random_forest(X_train, y_train):
    parameter = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 5, 10],
        "min_samples_split": [5, 10, 15],              
        "class_weight": ["balanced"],
        "criterion": ["gini", "entropy"]         
    }             

    random_forest = RandomForestClassifier(random_state=42)

    fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    rf_tuning = RandomizedSearchCV(estimator=random_forest, param_distributions=parameter, cv=fold, scoring="balanced_accuracy", verbose=2, n_jobs=-1)

    model = rf_tuning.fit(X_train, y_train)

    rf_best_model = model.best_estimator_
    return rf_best_model

def xgboost(X_train, y_train):
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    w = round(neg / pos, 2)

    parameter = {
        "n_estimators": [50, 100, 150],       
        "learning_rate": [0.01, 0.03, 0.05],            
        "max_depth": [3, 5, 7],                         
        "subsample": [0.8, 1.0],       
        "min_child_weight": [2, 3, 5, 10],    
        "class_weights": [w]               
    }

    xgb_model = XGBClassifier(random_state=42, tree_method="hist", eval_metric="logloss", n_jobs=-1)

    fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    xgb_tuning = RandomizedSearchCV(estimator=xgb_model, param_distributions=parameter, n_iter=30, cv=fold, scoring='balanced_accuracy', n_jobs=-1, verbose=2, random_state=42, return_train_score=True, refit=True)

    model = xgb_tuning.fit(X_train, y_train)

    xgb_best_model = model.best_estimator_ 
    return xgb_best_model

def catboost(X_train, y_train):
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    w = round(neg / pos, 2)

    parameter = {
        "iterations": [150, 200, 250],             
        "learning_rate": [0.01, 0.02, 0.03],         
        "depth": [3, 4, 5],                                           
        "subsample": [0.6, 0.7, 0.8],    
        "class_weights": [[1, w]]                       
    }

    cb = CatBoostClassifier(random_state=42, loss_function="Logloss",verbose=False,allow_writing_files=False)

    fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    cb_tuning = RandomizedSearchCV(estimator=cb, param_distributions=parameter, cv=fold, n_iter=30, scoring='balanced_accuracy', n_jobs=-1, verbose=2, random_state=42)

    model = cb_tuning.fit(X_train, y_train, verbose=True)

    cb_best_model = model.best_estimator_
    return cb_best_model

def balanced_xgboost(X_train, y_train):
    parameter = {
        "n_estimators": [50, 100, 150],       
        "learning_rate": [0.01, 0.03, 0.05],            
        "max_depth": [3, 5, 7],                         
        "subsample": [0.8, 1.0],       
        "min_child_weight": [2, 3, 5, 10],                   
    }

    xgb_model = XGBClassifier(random_state=42, tree_method="hist", eval_metric="logloss", n_jobs=-1)

    fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    xgb_tuning = RandomizedSearchCV(estimator=xgb_model, param_distributions=parameter, n_iter=30, cv=fold, scoring='balanced_accuracy', n_jobs=-1, verbose=2, random_state=42, return_train_score=True, refit=True)

    model = xgb_tuning.fit(X_train, y_train)

    xgb_best_model = model.best_estimator_ 
    return xgb_best_model

def balanced_catboost(X_train, y_train):
    parameter = {
        "iterations": [150, 200, 250],             
        "learning_rate": [0.01, 0.02, 0.03],         
        "depth": [3, 4, 5],                                           
        "subsample": [0.6, 0.7, 0.8],                         
    }

    cb = CatBoostClassifier(random_state=42, loss_function="Logloss",verbose=False,allow_writing_files=False)

    fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    cb_tuning = RandomizedSearchCV(estimator=cb, param_distributions=parameter, cv=fold, n_iter=30, scoring='balanced_accuracy', n_jobs=-1, verbose=2, random_state=42)

    model = cb_tuning.fit(X_train, y_train, verbose=True)

    cb_best_model = model.best_estimator_
    return cb_best_model