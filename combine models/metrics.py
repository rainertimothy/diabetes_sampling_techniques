from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score

def logistic_eval(model_logistic, X_train, y_train, X_test, y_test):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model_logistic, X_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
    accuracy_training = f"{cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}"

    y_pred_test = model_logistic.predict(X_test)
    accuracy = (y_test.values == y_pred_test).mean()

    clas_rep = classification_report(y_test, y_pred_test)

    precision = round(precision_score(y_test, y_pred_test, average="macro"), 2)
    recall = round(recall_score(y_test, y_pred_test, average="macro"), 2)
    f1 = round(f1_score(y_test, y_pred_test, average="macro"), 2)

    roc_auc = round(roc_auc_score(y_test, y_pred_test), 2)
    pr_auc = round(average_precision_score(y_test, y_pred_test), 2)

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.grid(False)
    
    return clas_rep, accuracy, precision, recall, f1, roc_auc, pr_auc, accuracy_training

def svm_eval(model_svm, X_train, y_train, X_test, y_test):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model_svm, X_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
    accuracy_training = f"{cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}"

    y_pred_test = model_svm.predict(X_test)
    accuracy = (y_test.values == y_pred_test).mean()

    clas_rep = classification_report(y_test, y_pred_test)

    precision = round(precision_score(y_test, y_pred_test, average="macro"), 2)
    recall = round(recall_score(y_test, y_pred_test, average="macro"), 2)
    f1 = round(f1_score(y_test, y_pred_test, average="macro"), 2)

    roc_auc = round(roc_auc_score(y_test, y_pred_test), 2)
    pr_auc = round(average_precision_score(y_test, y_pred_test), 2)

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.grid(False)

    return clas_rep, accuracy, precision, recall, f1, roc_auc, pr_auc, accuracy_training

def rf_eval(model_rf, X_train, y_train, X_test, y_test):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model_rf, X_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
    accuracy_training = f"{cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}"

    y_pred_test = model_rf.predict(X_test)
    accuracy = (y_test.values == y_pred_test).mean()

    clas_rep = classification_report(y_test, y_pred_test)

    precision = round(precision_score(y_test, y_pred_test, average="macro"), 2)
    recall = round(recall_score(y_test, y_pred_test, average="macro"), 2)
    f1 = round(f1_score(y_test, y_pred_test, average="macro"), 2)

    roc_auc = round(roc_auc_score(y_test, y_pred_test), 2)
    pr_auc = round(average_precision_score(y_test, y_pred_test), 2)

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.grid(False)
    
    return clas_rep, accuracy, precision, recall, f1, roc_auc, pr_auc, accuracy_training

def xgb_eval(model_xgb, X_train, y_train, X_test, y_test):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model_xgb, X_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
    accuracy_training = f"{cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}"

    y_pred_test = model_xgb.predict(X_test)
    accuracy = (y_test.values == y_pred_test).mean()

    clas_rep = classification_report(y_test, y_pred_test)

    precision = round(precision_score(y_test, y_pred_test, average="macro"), 2)
    recall = round(recall_score(y_test, y_pred_test, average="macro"), 2)
    f1 = round(f1_score(y_test, y_pred_test, average="macro"), 2)

    roc_auc = round(roc_auc_score(y_test, y_pred_test), 2)
    pr_auc = round(average_precision_score(y_test, y_pred_test), 2)

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.grid(False)
    
    return clas_rep, accuracy, precision, recall, f1, roc_auc, pr_auc, accuracy_training

def cb_eval(model_cb, X_train, y_train, X_test, y_test):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model_cb, X_train, y_train, cv=fold, scoring='balanced_accuracy', n_jobs=-1)
    accuracy_training = f"{cv_accuracy.mean():.2f} ± {cv_accuracy.std():.2f}"

    y_pred_test = model_cb.predict(X_test)
    accuracy = (y_test.values == y_pred_test).mean()

    clas_rep = classification_report(y_test, y_pred_test)

    precision = round(precision_score(y_test, y_pred_test, average="macro"), 2)
    recall = round(recall_score(y_test, y_pred_test, average="macro"), 2)
    f1 = round(f1_score(y_test, y_pred_test, average="macro"), 2)

    roc_auc = round(roc_auc_score(y_test, y_pred_test), 2)
    pr_auc = round(average_precision_score(y_test, y_pred_test), 2)

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.grid(False)
    
    return clas_rep, accuracy, precision, recall, f1, roc_auc, pr_auc, accuracy_training


