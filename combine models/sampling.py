import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

def adasyn_sampling(df_train):
    df_adasyn = df_train.copy()
    X = df_adasyn.drop("diabetic", axis=1)
    y = df_adasyn["diabetic"]
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

    df_adasyn = pd.DataFrame(X_adasyn, columns=X.columns)
    df_adasyn["diabetic"] = y_adasyn

    X_train = df_adasyn.drop("diabetic", axis=1)
    y_train = df_adasyn["diabetic"]

    return X_train, y_train

def smote_sampling(df_train):
    df_smote = df_train.copy()
    X = df_smote.drop("diabetic", axis=1)
    y = df_smote["diabetic"]
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    df_smote = pd.DataFrame(X_smote, columns=X.columns)
    df_smote["diabetic"] = y_smote

    X_train = df_smote.drop("diabetic", axis=1)
    y_train = df_smote["diabetic"]

    return X_train, y_train

def smotetomek_sampling(df_train):
    df_smotetomek = df_train.copy()
    X = df_smotetomek.drop("diabetic", axis=1)
    y = df_smotetomek["diabetic"]
    smotetomek = SMOTETomek(random_state=42)
    X_smotetomek, y_smotetomek = smotetomek.fit_resample(X, y)

    df_smotetomek = pd.DataFrame(X_smotetomek, columns=X.columns)
    df_smotetomek["diabetic"] = y_smotetomek

    X_train = df_smotetomek.drop("diabetic", axis=1)
    y_train = df_smotetomek["diabetic"]

    return X_train, y_train

def random_under_sampling(df_train):
    df_under = df_train.copy()
    X = df_under.drop("diabetic", axis=1)
    y = df_under["diabetic"]
    under = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_under, y_under = under.fit_resample(X, y)

    df_under = pd.DataFrame(X_under, columns=X.columns)
    df_under["diabetic"] = y_under

    X_train = df_under.drop("diabetic", axis=1)
    y_train = df_under["diabetic"]

    return X_train, y_train

def ratio_sampling(df_train, ratio):
    df_ratio = df_train.copy()
    x = df_ratio.drop("diabetic", axis=1)
    y = df_ratio["diabetic"]
    ratio = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    x_ratio, y_ratio = ratio.fit_resample(x,y)

    df_ratio = pd.DataFrame(x_ratio, columns=x.columns)
    df_ratio["diabetic"] = y_ratio

    X_train = df_ratio.drop("diabetic", axis=1)
    y_train = df_ratio["diabetic"]

    return X_train, y_train