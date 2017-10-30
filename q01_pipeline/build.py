import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.metrics import auc, roc_curve, accuracy_score

credit = pd.read_csv('data/credit_dataset_for_class.csv', index_col=0)
y = credit[['Student']]
X = credit.drop(['Student'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)


# Write your solution here
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


credit, _ = number_encode_features(credit)
credit.head()


def pipeline(X_train, X_test, y_train, y_test):
    smote = SMOTE(kind="regular", random_state=9)
    X_sm, y_sm = smote.fit_sample(X_train, y_train)
    logit = LogisticRegression(random_state=9)
    model = logit.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    return accuracy_score(y_test, y_pred), auc(fpr, tpr)
