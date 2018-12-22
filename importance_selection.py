import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
import lightgbm as lgb
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from hyperopt import hp, tpe
from model_class import ModelHandler
import read_data
import datetime
import pickle

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
METHOD = 'importance'
n_iter_search = 2

artificial_train, artificial_labes, artificial_test, digits_train, digits_labes, digits_test = read_data.read_data()

X_train = artificial_train
y = artificial_labes
X_test = artificial_test
dataset_name = 'artificial'

# X_train = digits_train
# y = digits_labes
# X_test = digits_test
# dataset_name = 'digits'

gbc = lgb.LGBMClassifier(is_unbalance=False, objective='binary', n_jobs=1, silent=True)

gbc.fit(X_train, y)

importances = gbc.feature_importances_

scores = []
variables = []
params = []

for importance_threshold in tqdm(range(max(importances) - 1)):

    X_train_filtered = X_train.iloc[:, importances > importance_threshold]
    X_test_filtered = X_test.iloc[:, importances > importance_threshold]

    if X_test.shape[1] > 0:

        gbc = lgb.LGBMClassifier(is_unbalance=False, objective='binary', n_jobs=1, silent=True)
        clf = RandomizedSearchCV(gbc,
                                 PARAMS,
                                 scoring='balanced_accuracy',
                                 n_iter=n_iter_search,
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=-1)

        clf.fit(X_train_filtered, y)

        gbc = lgb.LGBMClassifier(objective='binary', **clf.best_params_)

        score = cross_val_score(gbc, X_train_filtered, y, cv=5, scoring='balanced_accuracy')

        scores.append(np.mean(score))
        variables.append(X_train_filtered.columns.tolist())
        params.append(clf.best_params_)
    else:
        scores.append(-1)
        variables.append(tuple())
        params.append(tuple())

model = ModelHandler(params=params[np.argmax(scores)],
                     variables=variables[np.argmax(scores)],
                     score=max(scores),
                     dataset=dataset_name,
                     date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

with open('models/{}-{}-{}-{}-{}.pkl'.format(len(model.variables),
                                             round(model.score, 5),
                                             model.dataset,
                                             METHOD,
                                             model.date.replace(" ", '_')),
          'wb') as file:
    pickle.dump(model, file)

print("Best score: {}".format(max(scores)))
print("Used {} variables.".format(len(variables[np.argmax(scores)])))
