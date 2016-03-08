import numpy as np
from KaggleRandomForestTreeRegressor import KaggleRandomForestTreeRegressor


# INITIALIZATION

SEED_VALUE = 0
np.random.seed(SEED_VALUE)

X = np.genfromtxt('data/kaggle.X1.train.fragment.txt', delimiter=',')
Y = np.genfromtxt('data/kaggle.Y.train.fragment.txt', delimiter=',')
M, N = X.shape
K = KaggleRandomForestTreeRegressor(X, Y)


# EXECUTION

K.test_feature([i for i in range(1, 50, 5)], 'N_ESTIMATORS')
K.test_feature([i for i in range(1, N, 5)], 'MAX_FEATURES')
K.test_feature([i for i in range(1,20)], 'MAX_DEPTH')
K.test_feature([i for i in range(1, 20)], 'MIN_SAMPLES_SPLIT')
K.test_feature([i for i in range(1, 20)], 'MIN_SAMPLES_LEAF')
