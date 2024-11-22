import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nb import Model  # Assuming `Model` is implemented in the `nb` module

class TestSpotifyModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load dataset
        cls.data = pd.read_csv(r'.\spotify_discrete.csv')
        
        # Define features and target
        cls.X = cls.data.drop(columns=['liked'])
        cls.y = cls.data['liked']
        
        # Split data into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )
    
    def test_data_split_shapes(self):
        """Test if the data splits have the correct shapes."""
        self.assertEqual(self.X_train.shape[0], 0.8 * self.data.shape[0])
        self.assertEqual(self.X_test.shape[0], 0.2 * self.data.shape[0])
        self.assertEqual(self.X_train.shape[1], self.X.shape[1])

    def test_custom_model_training_and_testing(self):
        """Test the custom model training and testing workflow."""
        m = Model(self.X_train.reset_index(drop=True), self.y_train.reset_index(drop=True))
        
        # Train and test the custom model
        m.train()
        accuracy = m.test(self.X_test.reset_index(drop=True), self.y_test.reset_index(drop=True))
        
        # Check if accuracy is within valid range
        self.assertGreaterEqual(accuracy, 0.5)
        self.assertLessEqual(accuracy, 1.0)

    def test_sklearn_naive_bayes(self):
        """Test the sklearn Naive Bayes model."""
        # Instantiate the model
        nb_model = MultinomialNB()

        # Fit the model on training data
        nb_model.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = nb_model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

        # Ensure the classification report is non-empty
        report = classification_report(self.y_test, y_pred, output_dict=True)
        self.assertIn('accuracy', report)

if __name__ == "__main__":
    unittest.main()
