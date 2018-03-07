from unittest import TestCase
from ..build import pipeline, X_train, X_test, y_train, y_test, model
from inspect import getfullargspec
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy

bank_test = pd.read_csv('data/Bank_data_to_test.csv')
y = bank_test['y']
X = bank_test.drop(['y'], axis=1)
grid_model,  auc_score = pipeline(X_train, X_test, y_train, y_test, model)
prediction = grid_model.predict_proba(X)[:, 1]
auc_score_test = roc_auc_score(y, prediction)

class TestPipeline(TestCase):
    def test_pipeline_arguments(self):

        # Input parameters tests
        args = getfullargspec(pipeline)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))
    def test_pipeline_defaults(self):
        args = getfullargspec(pipeline)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types

    def test_pipeline_return_type_auc_score(self):    

        self.assertIsInstance(auc_score, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(auc_score)))

        # Return value tests
    def test_pipeline_return_value_auc_score(self):
        self.assertGreaterEqual(auc_score_test,numpy.float(0.72),"Expected value of Threshold auc not satisfied")
        # if numpy.float(0.72) <= auc_score_test <= numpy.float(1.00):
        #     self.assertTrue("You model has successfully passed the threshold auc value")
        # else:
        #     self.assertFalse("Expected value of Threshold auc not satisfied")
