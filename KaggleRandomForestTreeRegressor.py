import numpy as np
import matplotlib.pyplot as plt
from mltools import splitData
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
from sklearn import svm


class KaggleRandomForestTreeRegressor:
    def __init__(
        self,
        X=None,
        Y=None,
        params={},
        split_percentage=0.75,
        output_file='predictions.csv'
    ):
        if X is None:
            X = np.genfromtxt('data/kaggle.X1.train.fragment.txt', delimiter=',')
        if Y is None:
            Y = np.genfromtxt('data/kaggle.Y.train.fragment.txt', delimiter=',')

        self.Xtr, self.Xte, self.Ytr, self.Yte = splitData(X, Y, split_percentage)
        self.params = params
        self.output_file = output_file

    def set_params(self, new_params):
        self.params = new_params

    def _output_result_stats(self, feature_arr, error_arr):
        lowest_index, lowest_value = min(enumerate(error_arr), key=lambda p: p[1])
        highest_index, highest_value = max(enumerate(error_arr), key=lambda p: p[1])
        
        print 'lowest:', feature_arr[lowest_index], lowest_value
        print 'highest:', feature_arr[highest_index], highest_value
        
    def _draw_results(self, feature_arr, error_arr, feature_name):
        plt.plot(feature_arr, error_arr)
        plt.title(feature_name)
        plt.xlabel('Feature Value')
        plt.ylabel('Score (higher is better)')
        plt.show()

    def _create_random_forest(self, current_param={}):
        combined_param = dict(self.params, **current_param)
        clf = RandomForestRegressor()
        clf.set_params(**combined_param)
        clf = clf.fit(self.Xtr, self.Ytr)

        return clf

    def test_feature(self, feature_name, feature_arr):
        error_arr = []

        for feature in feature_arr:
            current_param = { feature_name: feature }
            clf = self._create_random_forest(current_param)
            score = clf.score(self.Xte, self.Yte)
            error_arr.append(score)

        self._draw_results(feature_arr, error_arr, feature_name)
        self._output_result_stats(feature_arr, error_arr)

    def write_predictions(self, X):
        clf = self._create_random_forest()
        Ye = clf.predict(X)

        fh = open(self.output_file,'w')
        fh.write('ID,Prediction\n')

        for i,yi in enumerate(Ye):
            fh.write('{},{}\n'.format(i+1,yi))

        fh.close()
