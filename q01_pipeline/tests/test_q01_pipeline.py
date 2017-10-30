from unittest import TestCase
from q01_pipeline.build import pipeline
from inspect import getargspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

credit = pd.read_csv('data/credit_dataset_for_class.csv', index_col=0)
y = credit[['Student']]
X = credit.drop(['Student'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


credit, _ = number_encode_features(credit)


class TestPipeline(TestCase):
    def test_pipeline(self):
        # Input parameters tests
        args = getargspec(pipeline)
        self.assertEqual(len(args[0]), 4, "Expected arguments %d, Given %d" % (4, len(args[0])))
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types
        accuracy_model, auc_score = pipeline(X_train, X_test, y_train, y_test)

        self.assertIsInstance(accuracy_model, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(accuracy_model)))

        self.assertIsInstance(auc_score, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(auc_score)))

        # Return value tests
        if auc_score >= 90 & auc_score <= 100:
            self.assertTrue("You model has successfully passed the threshold auc value")
        else:
            self.assertFalse("Expected value of Threshold auc not satisfied")