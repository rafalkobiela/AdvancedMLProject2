import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
import lightgbm as lgb
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from model_class import ModelHandler
import read_data
import datetime
import pickle

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features


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
METHOD = 'RFE'
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

# Create the RFE object and compute a cross-validated score.
gbc = lgb.LGBMClassifier(is_unbalance=False, objective='binary', n_jobs=1, silent=True)

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=gbc, step=1, cv=StratifiedKFold(5),
              scoring='balanced_accuracy')
rfecv.fit(X_train, y)

print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

X_train = X_train.iloc[:, rfecv.get_support()]



gbc = lgb.LGBMClassifier(is_unbalance=False, objective='binary', n_jobs=1, silent=True)
clf = RandomizedSearchCV(gbc,
                         PARAMS,
                         scoring='balanced_accuracy',
                         n_iter=n_iter_search,
                         cv=5,
                         n_jobs=-1,
                         verbose=-1)

clf.fit(X_train, y)

gbc = lgb.LGBMClassifier(objective='binary', **clf.best_params_)

score = cross_val_score(gbc, X_train, y, cv=5, scoring='balanced_accuracy')
score = np.mean(score)

model = ModelHandler(params=clf.best_params_,
                     variables=X_train.columns.tolist(),
                     score=score,
                     dataset=dataset_name,
                     date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

with open('models/{}-{}-{}-{}-{}.pkl'.format(len(model.variables),
                                             round(model.score, 5),
                                             model.dataset,
                                             METHOD,
                                             model.date.replace(" ", '_')),
          'wb') as file:
    pickle.dump(model, file)

print("Best score: {}".format(score))
print("Used {} variables.".format(len(model.variables)))
