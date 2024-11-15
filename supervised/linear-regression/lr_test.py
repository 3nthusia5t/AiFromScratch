import unittest
import numpy as np
from scipy.stats import zscore
from lr import Model  

# Asked ChatGPT to generate some tests for outlier removal.

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Initialize the Model object before each test
        self.model = Model(remove_outliers=True)

    def test_zscore_outlier_removal(self):
        # Test that `_zscore_outlier_removal` correctly filters outliers
        data = np.array([10, 12, 14, 15, 25, 14, 22, 10, 13, 19, 100])  # 100 is an outlier based on z-score
        filtered_data = self.model._zscore_outlier_removal(data)
        
        # 100 should be removed; check the output
        expected_result = np.array([10, 12, 14, 15, 25, 14, 22, 10, 13, 19])
        np.testing.assert_array_equal(filtered_data, expected_result)


if __name__ == '__main__':
    unittest.main()
