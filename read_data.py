import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_data():
    artificial_train = pd.read_csv('data/artificial_train.data', sep=' ', header=None).iloc[:, :-1]
    artificial_labes = pd.read_csv('data/artificial_train.labels', sep=' ', header=None)
    artificial_test = pd.read_csv('data/artificial_valid.data', sep=' ', header=None).iloc[:, :-1]

    scaler = StandardScaler()
    scaler.fit(artificial_train)
    artificial_train = pd.DataFrame(scaler.transform(artificial_train))
    artificial_test = pd.DataFrame(scaler.transform(artificial_test))

    artificial_train.columns = ['var_{}'.format(i) for i in range(artificial_train.shape[1])]
    artificial_labes.columns = ['y']
    artificial_test.columns = ['var_{}'.format(i) for i in range(artificial_test.shape[1])]

    digits_train = pd.read_csv('data/digits_train.data', sep=' ', header=None).iloc[:, :-1]
    digits_labes = pd.read_csv('data/digits_train.labels', sep=' ', header=None)
    digits_test = pd.read_csv('data/digits_valid.data', sep=' ', header=None).iloc[:, :-1]

    scaler = StandardScaler()
    scaler.fit(digits_train)
    digits_train = pd.DataFrame(scaler.transform(digits_train))
    digits_test = pd.DataFrame(scaler.transform(digits_test))

    digits_train.columns = ['var_{}'.format(i) for i in range(digits_train.shape[1])]
    digits_labes.columns = ['y']
    digits_test.columns = ['var_{}'.format(i) for i in range(digits_test.shape[1])]

    artificial_labes['y'].values

    return artificial_train, \
           artificial_labes['y'].values, \
           artificial_test, \
           digits_train, \
           digits_labes['y'].values, \
           digits_test
