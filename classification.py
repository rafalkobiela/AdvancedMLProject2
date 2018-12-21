import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
import lightgbm as lgb
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
import read_data
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.metrics import make_scorer, accuracy_score # change to balanced

PARAMS = {
    'learning_rate': uniform(0.1, 1),
    'num_leaves': sp_randint(20, 200),
    'max_depth': sp_randint(2, 10),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'min_child_weight': uniform(0, 0.1),
    'min_child_samples': sp_randint(10, 100),
    'feature_fraction': uniform(0.3, 0.7),
    'bagging_fraction': uniform(0.3, 0.7)
}

artificial_train, artificial_labes, artificial_test, digits_train, digits_labes, digits_test = read_data.read_data()

gbc = lgb.LGBMClassifier(is_unbalance=False, objective='binary', n_jobs=1, silent=True)

gbc.fit(artificial_train, artificial_labes)

scores = []

for importance_threshold in (range(max(gbc.feature_importances_))):

    artificial_train_filtered = artificial_train.iloc[:, gbc.feature_importances_ > importance_threshold]
    artificial_test_filtered = artificial_test.iloc[:, gbc.feature_importances_ > importance_threshold]

    if artificial_test_filtered.shape[1] > 0:

        n_iter_search = 20

        clf = RandomizedSearchCV(gbc,
                                 PARAMS,
                                 scoring='accuracy',
                                 n_iter=n_iter_search,
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=0)

        clf.fit(artificial_train_filtered, artificial_labes)

        gbc = lgb.LGBMClassifier(objective='binary', **clf.best_params_)

        score = cross_val_score(gbc, artificial_train_filtered, artificial_labes, cv=5, scoring='accuracy')

        scores.append(np.mean(score))
    else:
        scores.append(-1)



