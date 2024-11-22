import unittest
import numpy as np
from scipy.stats import zscore
from lr import Model  

class TestModel(unittest.TestCase):
    
    def setUp(self):
        self.model = Model(remove_outliers=True)

    def test_zscore_outlier_removal(self):
        data = np.array([10, 12, 14, 15, 25, 14, 22, 10, 13, 19, 100])
        filtered_data = self.model._zscore_outlier_removal(data)
        expected_result = np.array([10, 12, 14, 15, 25, 14, 22, 10, 13, 19])
        np.testing.assert_array_equal(filtered_data, expected_result)


if __name__ == '__main__':
    unittest.main()
