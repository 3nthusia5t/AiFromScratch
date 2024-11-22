import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, x_arr, y_arr, target_distribution_propotion=1, kernel='gauss'):
        """
        Initializes a Naive Bayes model.

        Parameters:
            df (pandas.DataFrame): the DataFrame to use for training the model
            problem (str): Either regression or classfication problem TODO: support regression
        """

        self.target = y_arr.to_numpy().reshape(-1, 2)
        self.features = x_arr.to_numpy()
        self.features_probs = []
        self.target_probs = []
        self.target_events = {}
        self.target_labels = None
        self.kernel = kernel
        self.target_and_features_probs = np.array([])
        

        # The proportion of unique values to all values of column, which determines if values are treated as discrete or continous.
        # Example: if proportion < 0.2, there is not a lot of unique values and we can treat it as discrete. 
        self.target_distibution_proportion = target_distribution_propotion
    
    def train(self):
        """
        Train the Naive Bayes model using the provided DataFrame.


        Returns:
        None
        """

        #targets probs has only one row, because there is only one feature - output
        self.target_events = self.__calculate_probs(self.target, self.target_distibution_proportion)[0]
        reshaped_target = self.target.reshape(-1,)

        # Build masks correctly
        masks = np.array([k == reshaped_target for k in self.target_events.keys()])
        self.features_given_e1 = self.features[masks[0]]
        self.features_given_e2 = self.features[masks[1]]

        #probabilities for each feature given e1
        self.feature_given_e1_probs = self.__calculate_probs(self.features_given_e1, self.target_distibution_proportion)
         #probabilities for each feature given e2
        self.feature_given_e2_probs = self.__calculate_probs(self.features_given_e2, self.target_distibution_proportion)


    def test(self, x_arr, y_arr):
        result = np.array([])
        for row in x_arr.to_numpy():
            e1_prob = 1
            e2_prob = 1
            for i in range(len(row)):
                try:
                    e1_prob *= self.feature_given_e1_probs[i][row[i]]
                except KeyError:
                    e1_prob *= 0.00001
                try:
                    e2_prob *= self.feature_given_e2_probs[i][row[i]]
                except KeyError:
                    e2_prob *= 0.00001
                    

            ks = []
            for k, _ in self.target_events.items():
                ks.append(k)

            e1_prob = e1_prob * self.target_events[ks[0]]
            e2_prob = e2_prob * self.target_events[ks[1]]
            
            test = e1_prob > e2_prob

            if e1_prob > e2_prob:
                result = np.append(result, ks[0])
            else:
                result = np.append(result, ks[1])

        good = 0
        y_arr = y_arr.to_numpy()
        for i in range(len(result)):
            if result[i] == y_arr[i]:
                good += 1
        return (good / len(result))





    def predict(self, x_arr):
        pass

    def __calculate_prob_feature_given_event(self, feature, label, event):
        pass

    def __calculate_probs(self, arr, proportion_limit, kernel_func=''):
        probs = np.array([])
        for col in arr.T:    
            if self.__is_discrete(col, proportion_limit):
                p, l= self.__calculate_discrete_probs(col)
                probs = np.append(probs,dict(map(reversed, zip(p,l))))
            else:
                print('naure')
                pass
        return probs
    
    def __calculate_discrete_probs(self, arr):
        labels, counts = np.unique(arr, return_counts=True)
        len_arr = len(arr)
        
    # Explicitly check if any value in (counts / len_arr) exceeds 1
        if np.any((counts / len_arr) > 1):
            raise Exception('Probability too high')
        return (counts / len(arr), labels) 

    def __is_discrete(self, arr, proportion_limit):
        value_count = len(arr)
        unique_count = len(np.unique(arr))
        return (unique_count / value_count) <= proportion_limit
