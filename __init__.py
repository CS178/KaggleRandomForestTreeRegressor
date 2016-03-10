import numpy as np
from KaggleRandomForestTreeRegressor import KaggleRandomForestTreeRegressor


# INITIALIZATION

SEED_VALUE = 0
np.random.seed(SEED_VALUE)

X = np.genfromtxt('data/kaggle.X1.train.fragment.txt', delimiter=',')
Y = np.genfromtxt('data/kaggle.Y.train.fragment.txt', delimiter=',')
M, N = X.shape

K = KaggleRandomForestTreeRegressor(X, Y)

# TESTING

params = {
    # 'max_features': 40,
    # 'max_depth': 15,
    # 'min_samples_split': 2,
    # 'min_samples_leaf': 11
}

K.set_params(params)

features_to_test = [
    ['n_estimators', [i for i in range(1, 50, 5)]],
    ['max_features', [i for i in range(1, N, 5)]],
    ['max_depth', [i for i in range(1, 20)]],
    ['min_samples_split', [i for i in range(1, 20)]],
    ['min_samples_leaf', [i for i in range(1, 20)]],
]

for feature in features_to_test:
    K.test_feature(feature[0], feature[1])


# PREDICTIONS

Xeval = np.genfromtxt('data/kaggle.X1.test.txt', delimiter=',')
params = {
    'n_estimators': 100,
    'max_features': 65,
    'max_depth': 12,
    'max_leaf_nodes': 45
}

K.set_params(params)
K.write_predictions(Xeval)
