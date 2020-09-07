import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def rm_corr(df, thresh):
    col = []
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_ls = [column for column in upper.columns if any(upper[column] > thresh)]
    return df.drop(drop_ls, axis=1)


def get_r_sq(X, y):
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    rss = sum((y - yhat) ** 2)
    tss = sum((y - np.mean(y)) ** 2)
    r_sq = 1 - rss / tss
    return r_sq


def get_iso_r_sq(df, y_label):
    result = []
    for column in df.drop(y_label, axis=1).columns:
        X = df.drop([column, y_label], axis=1)
        y = df[y_label]
        result.append(get_r_sq(X, y))
    return result


def get_impt(df, y_label):
    result = pd.Series(
        data=get_iso_r_sq(df, y_label), index=df.drop(y_label, axis=1).columns
    )
    r_sq = get_r_sq(df.drop(y_label, axis=1), df[y_label])
    result = result.apply(lambda x: r_sq - x)
    return result


def get_corr(df, y_label):
    result = df.corr()[y_label]
    return result
