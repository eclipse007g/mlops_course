import numpy as np
import unittest
import joblib
import pandas as pd

class IrisTest(unittest.TestCase):

    def test_sample(self):
        model = joblib.load("artifacts/model.joblib")
        data = pd.read_csv('sample/sample.csv')
        result = model.predict(data[['sepal_length','sepal_width','petal_length','petal_width']])
        print(result)
        self.assertEqual(result, "setosa", "Predicted class is wrong")

if __name__ == "__main__":
    unittest.main()