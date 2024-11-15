import threading
from scipy.stats import zscore
import numpy as np


class Model():
    
    def __init__(self, best_fit='err_expo', custom_best_fit = None):
        # Some config for model extensibility
        self.best_fit_func = {'err_sqr': self._best_fit_error_sqr, 'err_expo': self._best_fit_error_exponential}
        self.regression_line_dict = {'err_sqr' : lambda x: self.m*x+self.b, 'err_expo': lambda x: self.m/x + self.b}
        self.best_fit = self.best_fit_func[best_fit]
        self.regression_line = self.regression_line_dict[best_fit]
        if custom_best_fit != None:
            self.best_fit = custom_best_fit
        
        # Linear parameters
        self.m = 0
        self.b = 0
        self.accuracy = 0


    def remove_outliers(self, df, technique='zscore'):
        technique_to_func = {'zscore': self._zscore_outlier_removal}
        for column in df.columns:
            df = technique_to_func[technique](df, column)
        return df

    def train(self, x, y):
        assert len(x) == len(y)
        self.best_fit(x, y)



    def predict(self, x, y):
        self.accuracy = 0
        predictions = []
        for feature in x:
            predictions.append(self.regression_line(feature))


        mean = np.mean(y)
        sum_pd = 0
        sum_md = 0
        for i in range(len(y)):
            sum_pd += (y[i]-predictions[i])**2
            sum_md += (y[i]-mean)**2

        self.accuracy = 1 - (sum_pd/sum_md)
        return predictions

    def _best_fit_error_exponential(self, x, y):
        n = len(x)
        sum_ydivx = 0
        sum_1divx = 0
        sum_y = 0
        sum_1divx2 = 0
        for i in range(n):
            sum_ydivx += y[i] /x[i]
            sum_1divx += 1/x[i]
            sum_y += y[i]
            sum_1divx2 += 1/(x[i]**2)

        self.m = ((sum_ydivx*n)-(sum_y*sum_1divx))/((n*sum_1divx2)-(sum_1divx**2))
        self.b = ((sum_y*sum_1divx2)-(sum_ydivx*sum_1divx))/((n*sum_1divx2)-(sum_1divx**2))

    def _best_fit_error_sqr(self, x, y):
        n = len(x)
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        for i in range(n):
            sum_xy += x[i]*y[i]
            sum_x += x[i]
            sum_y += y[i]
            sum_x2 += x[i]**2
        
        self.m = ((n*sum_xy)-(sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
        self.b = ((sum_y*sum_x2)-(sum_xy*sum_x))/((n*sum_x2)-(sum_x**2))

    

    def error_square(self):
        pass


    '''
    zscore calculates how many standard deviations the value is from the mean.
    All the values that are 3 standard deviation away from the mean are considered outliers and removed.
    That can be tweaked using std_count parameter.
    '''
    def _zscore_outlier_removal(self, df, column_name, std_count=3):
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        col_data = df[column_name]
        z_scores = (col_data - col_data.mean()) / col_data.std(ddof=0)
        mask = np.abs(z_scores) < std_count
        return df[mask].reset_index(drop=True)