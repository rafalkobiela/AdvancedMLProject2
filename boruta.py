import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
import lightgbm as lgb
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from hyperopt import hp, tpe
from model_class import ModelHandler
import read_data
from boruta import BorutaPy
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
METHOD = 'boruta'
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


gbc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = BorutaPy(gbc, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(np.array(X_train), y)


X_train = X_train.iloc[:, feat_selector.support_]

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



