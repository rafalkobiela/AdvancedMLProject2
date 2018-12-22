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
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.metrics import balanced_accuracy_score


artificial_train, artificial_labes, artificial_test, digits_train, digits_labes, digits_test = read_data.read_data()

c_v = StratifiedKFold(3, random_state=21122018, shuffle=True) #initializing an instance of the estimator

clf = lgb.LGBMClassifier(
    n_jobs=-1,
    boosting_type='gbdt',
    objective='binary',
    metric='roc_auc',
    tree_learner='feature',
    silent=True,
    random_state=42,
    n_estimators=1000,
)

clf.fit(artificial_train, artificial_labes)


thresholds = np.sort(list(set(clf.feature_importances_)))

#initializing variables to hold controlling values
current_best_acc = np.NINF
current_best_variables = artificial_train.shape[1]
current_best_index = 0

for i, thresh in enumerate(thresholds):


    selection = SelectFromModel(clf, threshold=thresh, prefit=True)
    select_x_train = selection.transform(artificial_train)

    score = cross_val_score(clf,
                                select_x_train,
                                artificial_labes,
                                scoring='balanced_accuracy',
                                cv=c_v,
                                verbose=True,
                                n_jobs=-1
                               )

    rmse = score.mean()

    if rmse > current_best_acc:
        current_best_acc = rmse
        current_best_variables = select_x_train.shape[1]
        current_best_index = i
    else:
        continue

    print("Iteration {} out of {}".format(i, len(thresholds)))
    print("Thresh={}, n={}, RMSE: {}".format(thresh, select_x_train.shape[1], rmse))

    indices = np.argsort(clf.feature_importances_)[::-1]
    selected_indices = indices[0:current_best_variables]

with open('important_variables.txt', 'w') as file_handler:
    for item in selected_indices:
        file_handler.write("{}\n".format(item))


selected_indices = [105, 378, 318, 338, 442,  48, 241, 475, 453, 153, 336, 472]

trening = artificial_train.iloc[:, selected_indices]

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'colsample_bytree': params['colsample_bytree'],
        'subsample': params['subsample'],
        # 'n_estimators': int(params['n_estimators']),
        'num_leaves': int(params['num_leaves']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'max_bin': int(params['max_bin']),
        'bagging_freq': int(params['bagging_freq']),
        'learning_rate': params['learning_rate'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'min_sum_hessian_in_leaf': params['min_sum_hessian_in_leaf'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda']
    }

    x_lgb = lgb.LGBMClassifier(
    n_jobs=-1,
    boosting_type='gbdt',
    objective='binary',
    eval_metric='auc',
    tree_learner='feature',
    silent=True,
    random_state=42,
    n_estimators=1000,
    **params
    )
    score = cross_val_score(x_lgb, trening, artificial_labes, scoring='balanced_accuracy',
                            cv=StratifiedKFold(3, random_state=21122018, shuffle=True), verbose=True, n_jobs=-1)
    acc = score.mean()
    print("BACC {:.3f} params {}".format(acc, params))
    return acc


lgb_space = {
    'max_depth': hp.quniform('max_depth', 1, 45, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    # 'n_estimators': 250 + hp.randint('n_estimators', 800),
    'num_leaves': hp.quniform('num_leaves', 2, 200, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 200, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
    'max_bin': hp.quniform('max_bin', 32, 512, 1),
    'bagging_freq': hp.quniform('bagging_freq', 0, 5, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
    'reg_lambda': hp.uniform('reg_lambda', 0, 10)
}

best = fmin(fn=objective,
            space=lgb_space,
            algo=tpe.suggest,
            max_evals=512)

best['num_leaves'] = int(best['num_leaves'])
best['max_depth'] = int(best['max_depth'])
best['min_data_in_leaf'] = int(best['min_data_in_leaf'])
best['bagging_freq'] = int(best['bagging_freq'])
best['max_bin'] = int(best['max_bin'])
best['bagging_freq'] = int(best['bagging_freq'])


print("Hyperopt estimated optimum {}".format(best))

gbc = lgb.LGBMClassifier(
    n_jobs=-1,
    boosting_type='gbdt',
    objective='binary',
    metric='roc_auc',
    tree_learner='feature',
    silent=True,
    random_state=42,
    n_estimators=1000,
    **best
)

score = cross_val_score(gbc, trening, artificial_labes, cv=5, scoring='balanced_accuracy')
print(np.mean(score))
