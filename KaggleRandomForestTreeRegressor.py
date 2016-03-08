import numpy as np
import matplotlib.pyplot as plt
from mltools import splitData
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
from sklearn import svm


class KaggleRandomForestTreeRegressor:
    def __init__(self, X=None, Y=None):
        if X is None:
            X = np.genfromtxt('data/kaggle.X1.train.fragment.txt', delimiter=',')
        if Y is None:
            Y = np.genfromtxt('data/kaggle.Y.train.fragment.txt', delimiter=',')

        # split data set 75/25
        self.Xtr, self.Xte, self.Ytr, self.Yte = splitData(X, Y, 0.75)

    def _output_result_stats(self, feature_arr, error_arr):
        lowest_index, lowest_value = min(enumerate(error_arr), key=lambda p: p[1])
        highest_index, highest_value = max(enumerate(error_arr), key=lambda p: p[1])
        
        print 'lowest:', feature_arr[lowest_index], lowest_value
        print 'highest:', feature_arr[highest_index], highest_value
        
    def _draw_results(self, feature_arr, error_arr, feature_name):
        plt.plot(feature_arr, error_arr)
        plt.title(feature_name)
        plt.xlabel('Feature Value')
        plt.ylabel('Error')
        plt.show()

    def _create_random_forest(self, feature, feature_name):
        params_svm = {feature_name.lower(): feature}
        clf = RandomForestRegressor()
        clf.set_params(**params_svm)
        clf = clf.fit(self.Xtr, self.Ytr)

        return clf

    def test_feature(self, feature_arr, feature_name):
        error_arr = []

        for feature in feature_arr:
            clf = self._create_random_forest(feature, feature_name)
            score = clf.score(self.Xte, self.Yte)
            error_arr.append(score)

        self._draw_results(feature_arr, error_arr, feature_name)
        self._output_result_stats(feature_arr, error_arr)
