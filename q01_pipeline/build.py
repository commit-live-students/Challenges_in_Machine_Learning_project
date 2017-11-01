import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')

param_grid = dict(
    max_leaf_nodes=[50, 40, 60, 70],
    max_depth=[8, 10, 12, 15, 20],
    max_features=[4, 6, 8],
    min_samples_split=[50, 40, 30, 35]
)


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


bank, _ = number_encode_features(bank)

y = bank['y']
X = bank.drop(['y'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=9)


wt1, wt2 = compute_class_weight('balanced', np.unique(y_train), y=y_train)

model = RandomForestClassifier(random_state=9, oob_score=True, verbose=1, n_jobs=-1,
                               class_weight={0: wt1, 1: wt2},
                               n_estimators=1000)


def pipeline(X_train, X_test, y_train, y_test, model, param_grid):
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid.fit(X_train, y_train)
    y_pred = grid.predict_proba(X_test)[:, 1]
    return grid, roc_auc_score(y_test, y_pred)

