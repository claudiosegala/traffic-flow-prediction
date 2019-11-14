# -*- coding: utf-8 -*-

import sklearn
import time
import random
import copy

import pandas as pd # data manipulation library
import numpy as np # math library

import sklearn.metrics as sklm # metrics
import statsmodels as sm # statistical models

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

"""## Configurations

Make the environment reproducible
"""

import tensorflow as tf # machine learning library
import os

os.environ['PYTHONHASHSEED'] = '0'
tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(0)
random.seed(0)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""Set path for the folder in which everything should be stored."""

PATH = ''

"""## Util"""

class WalkingForwardTimeSeriesSplit():
  def __init__(self, n_splits, skip = 0, take = 100000000):
    self.n_splits = n_splits
    self.skip = min(skip, n_splits)
    self.take = min(take, n_splits)
  
  def get_n_splits(self, X, y, groups):
    return min(self.skip + self.take, self.n_splits) - self.skip
  
  def split(self, X, y=None, groups=None):
    n_samples = len(X)
    k_fold_size = n_samples // self.n_splits
    indices = np.arange(n_samples)

    margin = 0
    for i in range(self.skip, min(self.skip + self.take, self.n_splits)):
      start = i * k_fold_size
      stop = start + k_fold_size
      mid = int(0.8 * (stop - start)) + start
      yield indices[start: mid], indices[mid + margin: stop]

def clean_cv_results(cv_results):
  res = copy.deepcopy(cv_results)
  for key in res:
    if type(res[key]) != list: 
      res[key] = res[key].tolist()

  return res

"""## Dataset Generation Util"""

def retrieve_data(flow_interval):
    path = "{0}dataset/dataset_flow_{1}.csv".format(PATH, flow_interval)
    print(PATH)
    data = pd.read_csv(path, ';')
    
    data['Flow'].apply(int)
    data['AveSpeed'].apply(float)
    data['Density'].apply(float)
    data['Sunday'].apply(int)
    data['Monday'].apply(int)
    data['Tuesday'].apply(int)
    data['Wednesday'].apply(int)
    data['Thursday'].apply(int)
    data['Friday'].apply(int)
    data['Saturday'].apply(int)
      
    return data

"""## Storage Util"""

import json

def print_json (obj):
  print(json.dumps(obj, sort_keys=True, indent=4))

def store(obj, path, name):
  with open("{0}{1}/{2}.json".format(PATH, path, name), 'w') as json_file:
    json.dump(obj, json_file, sort_keys=True, indent=4)

def store_results ():
  name = int(time.time())
  
  result_data['meta'] = {
    "SEEABLE_PAST": SEEABLE_PAST,
    "PREDICT_IN_FUTURE": PREDICT_IN_FUTURE,
    "FLOW_INTERVAL": FLOW_INTERVAL,
    "N_SPLITS": N_SPLITS,
  }

  store(result_data, "results", name)

  slim_result_data = copy.deepcopy(result_data)
  for model in slim_result_data['results']:
      del slim_result_data['results'][model]['raw']

  store(slim_result_data, "results", "{0}_slim".format(name))

def store_comparisons (title):
  name = str(int(time.time()))
  
  j = copy.deepcopy(comparison_data)

  store(j, "results/comparison", "{0}_{1}".format(title, name))
    
  for i in range(len(j)):
    for model in j[i]['results']:
      del j[i]['results'][model]['raw']

  store(j, "results/comparison", "{0}_{1}_slim".format(title, name))

"""## Models Util

### Dropped

#### Random (Baseline)

This implementation just guess a random number in the [0, 100] interval for every output.
"""

def random_guess_univariate (data):
  global result_data
  
  X, Y = generate_dataset(data, False, FLOW_INTERVAL, N_STEPS, N_FUTURE)

  name = "Random Guess"
  m = max(Y)

  expected, observed, times = [], [], []
  pointers = split_dataset(len(Y), SET_SPLIT, TEST_SPLIT)
  
  for i, j, k in pointers:
    start = time.time()

    Y_hat = [random.randint(0, m) for i in range(k - j)]

    expected.append(Y[j:k])
    observed.append(Y_hat)
    times.append(time.time() - start)

  result_data['results'][name] = evaluate(expected, observed, times, name)

"""#### ARIMA

This implementation was based on [How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/).
"""

def arima (X):
  size = int(len(X) * TRAIN_SPLIT)
  acc = X[size-(2*WEEK_SIZE):size]
  Y = X[size+N_FUTURE:]
  Y_hat = []
  
  #for t in range(len(Y)):
  for t in range(50):
    print(t, len(Y))
    
    model = sm.tsa.arima_model.ARIMA(acc, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    
    start = len(acc)
    end = start + N_FUTURE
    
    prediction = model_fit.predict(start=start, end=end+1)
    
    print(prediction)
    Y_hat.append(prediction[-1])    
    acc.append(X[size + t])
    acc.pop(0)
  
  print_difference(Y, Y_hat)

"""#### Logistic Regression"""

from sklearn.linear_model import LogisticRegression

def logistic_regression(data, useB):
  global result_data
  
  name = "LR B" if useB else "LR A"

  expected, observed, times = [], [], []

  X, Y = generate_dataset(data, useB, FLOW_INTERVAL, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

  model = LogisticRegression()

  pointers = split_dataset(len(X), SET_SPLIT, TEST_SPLIT)
  
  for i, j, k in pointers:
    start = time.time()
    
    model.fit(X[i:j], Y[i:j])
    
    expected.append(Y[j:k])
    observed.append(model.predict(X[j:k]))
    times.append(time.time() - start)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

"""#### RNN

The optimzation was based on [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/).
"""

from keras.layers import SimpleRNN

def rnn (data, useB): 
  global result_data
  
  name = "RNN B" if useB else "RNN A"
  
  X, Y = generate_dataset(data, useB, FLOW_INTERVAL, N_STEPS, N_FUTURE)
  
  expected, observed, times = [], [], []
  
  model = Sequential()		

  model.add(SimpleRNN(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))		
  model.add(Dense(1))		

  model.compile(optimizer='adam', loss='mse', metrics = ["accuracy"])
  
  pointers = split_dataset(len(X), SET_SPLIT, TEST_SPLIT)
  
  for i, j, k in pointers:
    start = time.time()
    
    model.fit(X[i:j], Y[i:j], validation_split=0.2, batch_size=64, epochs=15, verbose=0)
    
    expected.append(Y[j:k])
    observed.append(model.predict(X[j:k]))
    times.append(time.time() - start)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

"""### Misc

Function to help implement the training and evaluation of the models.
"""

def evaluate_precision_hit_ratio (Y, Y_hat):
  """ Trend Prediction Ratio Calculation
  
  Calculates the ratio of up/down prediction.
  
  Arguments:
    Y: the expected dataset.
    Y_hat: the observed dataset.
  """
  
  cnt = 0
  
  for i in range(len(Y)):
    if i < N_FUTURE:
      continue
      
    exp = Y[i] - Y[i - N_FUTURE]
    obs = Y_hat[i] - Y[i - N_FUTURE]
    
    if exp * obs > 0:
      cnt += 1
    
  return cnt / len(Y)

def evaluate_precision_bucket (Y, Y_hat):
  """ Precision Bucket Calculation
  
  Counts how many of the prediction got wronng by at most 2ˆx, x 
  being the bucket. There are 7 buckets, that is, the maximum error 
  calculated is 128.
  
  Arguments:
    Y: the expected dataset.
    Y_hat: the observed dataset.
  """
  
  n = 7 # the number of buckets
  buckets = [0] * n
  
  for i in range(len(Y)):
    diff = abs(Y[i] - Y_hat[i])
    
    for i in range (n):
      if diff <= 2**i:
        buckets[i] += 1
        break

  for i in range (n):
     buckets[i] = buckets[i] / len(Y)

  return tuple(buckets)

def evaluate_raw (expected, observed, times):
  """ Evaluate Raw Sessions 
  
  Evaluate each of the train&test sessions by RMSE, NRMSE, MAE, HR, PRE. 
  It will store the results in a object and return it.
  
  Arguments:
    expected: an array of expected instances of each train&test session.
    observed: an array of observed instances of each train&test session.
    times: an array of the time of each train&test session.
  """
  
  n = len(expected)

  for i in range(n):
    observed[i] = [0 if np.isnan(o) else o for o in observed[i]]

  for i in range(n):
    observed[i] = [max(o, 0) for o in observed[i]]
  
  raw = {
    'expected': expected,
    'observed': observed,
    'TIME': times,
    'RMSE': [0] * n,
    # 'NRMSE': [0] * n,
    'MAE': [0] * n,
    'HR': [0] * n,
    #'PRE': [0] * n,
  }
  
  for i in range(n):
    Y = expected[i]
    Y_hat = observed[i]
    time = times[i]

    raw['MAE'][i] = sklm.mean_absolute_error(Y, Y_hat)
    raw['RMSE'][i] = np.sqrt(sklm.mean_squared_error(Y, Y_hat))
    # raw['NRMSE'][i] = raw['RMSE'][i] / np.std(Y)
    raw['HR'][i] = evaluate_precision_hit_ratio(Y, Y_hat)
    #raw['PRE'][i] = evaluate_precision_bucket(Y, Y_hat)
    
    if VERBOSITY:
      print("({0}/{1}) Test Size: {2}, Time: {3}s".format(i+1, n, len(Y), time))
      print("\tRMSE: {0}".format(raw['RMSE'][i]))
      # print("\tNRMSE: {0}".format(raw['NRMSE'][i]))
      print("\tMAE: {0}".format(raw['MAE'][i]))
      print("\tHit Ratio: {0}%".format(raw['HR'][i] * 100))

  return raw

def evaluate (expected, observed, times, name):
  """ Evaluate Sessions
  
  Evaluate models by RMSE, NRMSE, MAE, HR, PRE. It will store the 
  results in a object and return it.
  
  Arguments:
    expected: an array of expected instances of each 
      train&test session.
    observed: an array of observed instances of each 
      train&test session.
    times: an array of the time of each train&test session.
    name: the name of the model
  """
  n = len(expected)
  flatten = lambda l : [i for sl in l for i in sl]
  
  # Make the arrays serializable
  expected = list(map(list, expected))
  observed = list(map(list, observed))
  
  for i in range(n):
    expected[i] = list(map(float, expected[i]))
    observed[i] = list(map(float, observed[i]))
  
  raw = evaluate_raw(expected, observed, times)
  
  #n_buckets = len(raw['PRE'])
  #_pre = [[pre[i] for pre in raw['PRE']] for i in range(n_buckets)]
  
  eva = {
    'TIME': int(sum(times)),
    'RMSE': float(np.mean(raw['RMSE'])),
    # 'NRMSE': float(np.mean(raw['NRMSE'])),
    'MAE': float(np.mean(raw['MAE'])),
    'HR': float(np.mean(raw['HR'])),
    #'PRE': [float(np.mean(p)) for p in _pre],
    'has_negative': (min(flatten(observed)) < 0),
    'raw': raw
  }
  
  print("\n{0} Final Result:".format(name))
  print("\tTotal Time: {0}s".format(eva['TIME']))
  print("\tRMSE: {0}".format(eva['RMSE']))
  # print("\tNRMSE: {0}".format(eva['NRMSE']))
  print("\tMAE: {0}".format(eva['MAE']))
  print("\tHit Ratio: {0}%".format(eva['HR'] * 100))
  #print("\tPrecision: {0}".format(eva['PRE']))
    
  return eva

def generate_dataset(data, useB, n_steps, n_future):
  """ Generate Dataset
  
  Generate a dataset provided a sequence. Reshape the sequence in rolling intervals from [samples, timesteps] into 
  [samples, timesteps, features] and split the sequence. The split the sequence in rolling intervals with a corresponding value 
  like the example bellow.

  Ex: split_sequence([1, 2, 3, 4, 5], 3) #([[1, 2, 3], [2, 3, 4]], [4, 5])
  
  Arguments:
    raw_seq: the sequence to reshape.
    useB: if the dataset is more complex or not.
    n_steps: size of the rolling interval
    n_future: the distance to the interval the value should be.  
  """

  sequence = np.array(data if useB else data['Flow'])

  n = len(sequence)
  X, Y = list(), list()

  for i in range(n):
    j = i + n_steps
    k = j + n_future

    if k >= n:
      break

    seq_x, seq_y = sequence[i:j], sequence[k]
    X.append(seq_x)	
    Y.append(seq_y[0] if useB else seq_y)

  X, Y = np.array(X), np.array(Y)	
  
  if not useB:
    X = X.reshape((X.shape[0], X.shape[1], 1))

  return X, Y

"""### Moving Average (Baseline)

This implementation just get the mean of every flow value in the input and place it as output.
"""

def moving_average (data):
  global result_data

  name = "Moving Average"
  
  X, Y = generate_dataset(data, False, N_STEPS, N_FUTURE)

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)
  expected, observed, times = [], [], []

  for train_index, test_index in cv.split(X):
    X_test = X[test_index]
    Y_test = Y[test_index]
  
    start_time = time.time()
    Y_hat = [np.mean(x) for x in X_test]
    end_time = time.time()
    
    expected.append(Y_test)
    observed.append(Y_hat)
    times.append(end_time - start_time)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

"""### Naive (Baseline)

This implementation just use the last value of input as output.
"""

def naive (data):
  global result_data

  name = "Naive"
  
  X, Y = generate_dataset(data, False, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1])

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)
  expected, observed, times = [], [], []

  for train_index, test_index in cv.split(X):
    X_test = X[test_index]
    Y_test = Y[test_index]
  
    start_time = time.time()
    Y_hat = [x[-1] for x in X_test]
    end_time = time.time()
    
    expected.append(Y_test)
    observed.append(Y_hat)
    times.append(end_time - start_time)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

"""### Random Forest

This implementation is based on [Random Forest Algorithm with Python and Scikit-Learn](https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/)
"""

from sklearn.ensemble import RandomForestRegressor

def random_forest_grid(data, useB):
  global result_data

  name = "RF Grid B" if useB else "RF Grid A"
  
  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    
  param_grid = {
    # 'bootstrap': [True, False],
    'max_depth': [8, 16, 32, 64, None],
    'n_estimators': [50, 100, 200, 400, 800],
  }
  model = sklearn.ensemble.RandomForestRegressor(max_features='auto', random_state=0)
  scoring = 'neg_mean_squared_error'
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, take=N_SPLITS-1)
  n_jobs = 8 if useB else 2
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=tscv, n_jobs=n_jobs, verbose=2)

  start_time = time.time()
  grid_search.fit(X, Y)
  end_time = time.time()

  best_model = grid_search.best_estimator_

  expected, observed, times = [], [], []

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, skip=N_SPLITS-1)

  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    _start_time = time.time()
    best_model.fit(X_train, Y_train)
    _end_time = time.time()
    
    expected.append(Y_test)
    observed.append(best_model.predict(X_test))
    times.append(_end_time - _start_time)
    
  res = evaluate(expected, observed, times, name)
  res['time'] = end_time - start_time
  res['best_params'] = grid_search.best_params_
  res['params'] = param_grid
  res['score'] = clean_cv_results(grid_search.cv_results_)

  store(res, "results/grid", name)

def random_forest(data, useB):
  global result_data
  
  name = "RF B" if useB else "RF A"
  
  model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_features='auto', random_state=0)

  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

  expected, observed, times = [], [], []
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)

  for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    start_time = time.time()
    model.fit(X_train, Y_train)
    end_time = time.time()
    
    expected.append(Y_test)
    observed.append(model.predict(X_test))
    times.append(end_time - start_time)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

def random_forest_tune(X, Y, useB):
  cv = WalkingForwardTimeSeriesSplit(n_splits=1)

  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # param_grid = {
    #   # 'bootstrap': [True, False],
    #   'max_depth': [8, 16, 32, 64, None],
    #   'n_estimators': [50, 100, 200, 400, 800],
    # }
    param_grid = {
      # 'bootstrap': [True, False],
      'max_depth': [4, 8],
      'n_estimators': [15],
    }
    model = sklearn.ensemble.RandomForestRegressor(max_features='auto', random_state=0)
    scoring = 'neg_mean_squared_error'
    cv = [(slice(None), slice(None))]
    n_jobs = 8 if useB else 2

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=2)

    grid_search.fit(X_train, Y_train)
      
    res = {
        'params': param_grid,
        'best_params': grid_search.best_params_,
        'score': clean_cv_results(grid_search.cv_results_),
    }

    return grid_search.best_estimator_, res

def random_forest_2(data, useB):
  global result_data
  
  name = "RF B" if useB else "RF A"
  
  model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_features='auto', random_state=0)

  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

  expected, observed, times, grid_res = [], [], [], []
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)

  for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    start_time = time.time()
    tuned_model, res = random_forest_tune(X_train, Y_train, useB)
    tuned_model.fit(X_train, Y_train) 
    end_time = time.time()
    
    grid_res.append(res)
    expected.append(Y_test)
    observed.append(tuned_model.predict(X_test))
    times.append(end_time - start_time)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)
  store(result_data['results'][name], "results/grid", name)

"""### Support Vector Machine"""

from sklearn import svm

def support_vector_machine_grid(data, useB):
  global result_data

  name = "SVM Grid B" if useB else "SVM Grid A"
  
  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    
  param_grid = {
    'C': [1.0, 10.0, 100.0, 1000.0],
    'gamma': list(np.logspace(-2, 2, 4)) + ['scale'],
    # 'epsilon': [0.01, 0.1, 1]
  }
  model = svm.SVR(epsilon=1.0)
  scoring = 'neg_mean_squared_error'
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, take=N_SPLITS-1)
  n_jobs = 8 if useB else 2
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=tscv, n_jobs=n_jobs, verbose=2)

  start_time = time.time()
  grid_search.fit(X, Y)
  end_time = time.time()

  best_model = grid_search.best_estimator_

  expected, observed, times = [], [], []

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, skip=N_SPLITS-1)

  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    _start_time = time.time()
    best_model.fit(X_train, Y_train)
    _end_time = time.time()
    
    expected.append(Y_test)
    observed.append(best_model.predict(X_test))
    times.append(_end_time - _start_time)

  res = evaluate(expected, observed, times, name)
  res['time'] = end_time - start_time
  res['best_params'] = grid_search.best_params_
  res['params'] = param_grid
  res['score'] = clean_cv_results(grid_search.cv_results_)

  store(res, "results/grid", name)

def support_vector_machine(data, useB):
  global result_data
  
  name = "SVM B" if useB else "SVM A"
  
  model = svm.SVR(gamma='scale', C=1.0, epsilon=0.2)

  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)
  X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

  expected, observed, times = [], [], []
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)

  for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    start_time = time.time()
    model.fit(X_train, Y_train)
    end_time = time.time()
    
    expected.append(Y_test)
    observed.append(model.predict(X_test))
    times.append(end_time - start_time)
    
  result_data['results'][name] = evaluate(expected, observed, times, name)

"""### LSTM"""

from keras.layers import LSTM

def create_lstm(input_shape):
  def create(n=100, activation='sigmoid'):
    model = Sequential()		

    model.add(LSTM(n, activation=activation, input_shape=input_shape))		
    model.add(Dense(1))		

    model.compile(optimizer='adam', loss='mse', metrics = ["accuracy"])	

    return model

  return create

def lstm_grid(data, useB):		
  global result_data

  name = "LSTM Grid B" if useB else "LSTM Grid A"
        
  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)

  param_grid = {		
    'n': [50, 100, 200, 400, 800],
    'batch_size': [8, 16, 32, 64]
  }
  model = KerasClassifier(build_fn=create_lstm((X.shape[1], X.shape[2])), validation_split=0.2, epochs=15, verbose=0)
  scoring = 'neg_mean_squared_error'
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, take=N_SPLITS-1)
  n_jobs = 8 if useB else 2
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=tscv, n_jobs=n_jobs, verbose=2)
      
  start_time = time.time()
  grid_search.fit(X, Y)
  end_time = time.time()

  best_model = grid_search.best_estimator_

  expected, observed, times = [], [], []

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, skip=N_SPLITS-1)

  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    _start_time = time.time()
    best_model.fit(X_train, Y_train)
    _end_time = time.time()
    
    expected.append(Y_test)
    observed.append(best_model.predict(X_test))
    times.append(_end_time - _start_time)

  res = evaluate(expected, observed, times, name)
  res['time'] = end_time - start_time
  res['best_params'] = grid_search.best_params_
  res['params'] = param_grid
  res['score'] = clean_cv_results(grid_search.cv_results_)

  store(res, "results/grid", name)

def lstm (data, useB): 
  global result_data
  
  name = "LSTM B" if useB else "LSTM A"

  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)

  model = create_lstm((X.shape[1], X.shape[2]))()
  
  expected, observed, times, hist = [], [], [], []
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)

  for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
  
    start_time = time.time()
    h = model.fit(X_train, Y_train, validation_split=0.2, batch_size=64, epochs=15, verbose=0)
    end_time = time.time()

    expected.append(Y_test)
    observed.append(model.predict(X_test))
    times.append(end_time - start_time)
    hist.append(h)

  result_data['results'][name] = evaluate(expected, observed, times, name)
  result_data['results'][name]['history'] = {
      'loss': [h.history['loss'] for h in hist],
      'val_loss': [h.history['val_loss'] for h in hist],
  }

"""### GRU"""

from keras.layers import GRU

def create_gru(input_shape):
  def create(n=100, activation='sigmoid'):
    model = Sequential()		

    model.add(GRU(n, activation=activation, input_shape=input_shape))		
    model.add(Dense(1))		

    model.compile(optimizer='adam', loss='mse', metrics = ["accuracy"])		

    return model		
    
  return create

def gru_grid(data, useB):		
  global result_data

  name = "GRU Grid B" if useB else "GRU Grid A"
        
  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)

  param_grid = {		
    'n': [50, 100, 200, 400, 800],
    'batch_size': [8, 16, 32, 64]
  }
  model = KerasClassifier(build_fn=create_gru((X.shape[1], X.shape[2])), validation_split=0.2, epochs=15, verbose=0)
  scoring = 'neg_mean_squared_error'
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, take=N_SPLITS-1)
  n_jobs = 8 if useB else 2
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=tscv, n_jobs=n_jobs, verbose=2)		

  start_time = time.time()
  grid_search.fit(X, Y)
  end_time = time.time()

  best_model = grid_search.best_estimator_

  expected, observed, times = [], [], []

  cv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS, skip=N_SPLITS-1)

  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    _start_time = time.time()
    best_model.fit(X_train, Y_train)
    _end_time = time.time()
    
    expected.append(Y_test)
    observed.append(best_model.predict(X_test))
    times.append(_end_time - _start_time)

  res = evaluate(expected, observed, times, name)
  res['time'] = end_time - start_time
  res['best_params'] = grid_search.best_params_
  res['params'] = param_grid
  res['score'] = clean_cv_results(grid_search.cv_results_)

  store(res, "results/grid", name)

def gru (data, useB): 
  global result_data
  
  name = "GRU B" if useB else "GRU A"

  X, Y = generate_dataset(data, useB, N_STEPS, N_FUTURE)

  model = create_gru((X.shape[1], X.shape[2]))()
  
  expected, observed, times, hist = [], [], [], []
  tscv = WalkingForwardTimeSeriesSplit(n_splits=N_SPLITS)

  for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
  
    start_time = time.time()
    h = model.fit(X_train, Y_train, validation_split=0.2, batch_size=64, epochs=15, verbose=0)
    end_time = time.time()

    expected.append(Y_test)
    observed.append(model.predict(X_test))
    times.append(end_time - start_time)
    hist.append(h)
  
  result_data['results'][name] = evaluate(expected, observed, times, name)
  result_data['results'][name]['history'] = {
      'loss': [h.history['loss'] for h in hist],
      'val_loss': [h.history['val_loss'] for h in hist],
  }

"""## Comparison Util"""

def run_models():
  global result_data
  
  result_data = {
      'results': {},
      'meta': {
        'SEEABLE_PAST': SEEABLE_PAST,
        'PREDICT_IN_FUTURE': PREDICT_IN_FUTURE,
        'FLOW_INTERVAL': FLOW_INTERVAL,
        'N_SPLITS': N_SPLITS,
      }
  }

  data = retrieve_data(FLOW_INTERVAL)

  moving_average(data)
  naive(data)
  random_forest(data, False)
  random_forest(data, True)
  support_vector_machine(data, False)
  support_vector_machine(data, True)
  lstm(data, False)
  lstm(data, True)
  gru(data, False)
  gru(data, True)

  store_results()

def compare_results_by_n_split(values):
  global N_SPLITS
  global comparison_data
  
  aux = N_SPLITS
  comparison_data = []
  
  for value in values:
    N_SPLITS = value

    start_time = time.time()
    run_models()
    end_time = time.time()
    
    comparison_data.append(copy.deepcopy(result_data))

    print("({0} of {1}) Finished Running with N_SPLITS {2} in {3} seconds".format(len(comparison_data), len(values), value, end_time - start_time))

  store_comparisons('n_split_comparison')
  
  N_SPLITS = aux

def compare_results_by_seeable_past(values):
  global SEEABLE_PAST
  global N_STEPS
  global comparison_data
  
  aux = SEEABLE_PAST
  comparison_data = []
  
  for value in values:
    SEEABLE_PAST = value
    N_STEPS = SEEABLE_PAST * 60 // FLOW_INTERVAL

    start_time = time.time()
    run_models()
    end_time = time.time()
    
    comparison_data.append(copy.deepcopy(result_data))

    print("({0} of {1}) Finished Running with SEEABLE_PAST {2} in {3} seconds".format(len(comparison_data), len(values), value, end_time - start_time))

  store_comparisons('seeable_past_comparison')

  SEEABLE_PAST = aux
  N_STEPS = SEEABLE_PAST * 60 // FLOW_INTERVAL

def compare_results_by_flow_interval(values):
  global FLOW_INTERVAL
  global N_STEPS
  global N_FUTURE
  global DAY_SIZE
  global WEEK_SIZE
  global comparison_data
  
  aux = FLOW_INTERVAL
  comparison_data = []
  
  for value in values:
    FLOW_INTERVAL = value
    N_STEPS = SEEABLE_PAST * 60 // FLOW_INTERVAL
    N_FUTURE = PREDICT_IN_FUTURE * 60 // FLOW_INTERVAL
    DAY_SIZE = (24 * 3600) // FLOW_INTERVAL  
    WEEK_SIZE = 7 * DAY_SIZE

    start_time = time.time()
    run_models()
    end_time = time.time()
    
    comparison_data.append(copy.deepcopy(result_data))

    print("({0} of {1}) Finished Running with FLOW_INTERVAL {2} in {3} seconds".format(len(comparison_data), len(values), value, end_time - start_time))

  store_comparisons('flow_interval_comparison')
  
  FLOW_INTERVAL = aux
  N_STEPS = SEEABLE_PAST * 60 // FLOW_INTERVAL
  N_FUTURE = PREDICT_IN_FUTURE * 60 // FLOW_INTERVAL
  DAY_SIZE = (24 * 3600) // FLOW_INTERVAL  
  WEEK_SIZE = 7 * DAY_SIZE

def compare_results_by_predict_in_future(values):
  global PREDICT_IN_FUTURE
  global N_FUTURE
  global comparison_data
  
  aux = PREDICT_IN_FUTURE
  comparison_data = []
  
  for value in values:
    PREDICT_IN_FUTURE = value
    N_FUTURE = PREDICT_IN_FUTURE * 60 // FLOW_INTERVAL

    start_time = time.time()
    run_models()
    end_time = time.time()
    
    comparison_data.append(copy.deepcopy(result_data))

    print("({0} of {1}) Finished Running with PREDICT_IN_FUTURE {2} in {3} seconds".format(len(comparison_data), len(values), value, end_time - start_time))

  store_comparisons('predict_future_comparison')
  
  PREDICT_IN_FUTURE = aux
  N_FUTURE = PREDICT_IN_FUTURE * 60 // FLOW_INTERVAL

"""## Train&Test

Run all the models and store the results at the end
"""

# Model Parameters
SEEABLE_PAST = 480 # in minutes
PREDICT_IN_FUTURE = 60 # in minutes
FLOW_INTERVAL = 150 # the interval size for each flow
N_SPLITS = 8

# Derivated Model Parameters
N_STEPS = SEEABLE_PAST * 60 // FLOW_INTERVAL # the number of flows to see in the past
N_FUTURE = PREDICT_IN_FUTURE * 60 // FLOW_INTERVAL # how much in the future we want to predict (0 = predict the flow on the next FLOW_INTERVAL minutes)
DAY_SIZE = (24 * 60 * 60) // FLOW_INTERVAL  
WEEK_SIZE = (7 * 24 * 60 * 60) // FLOW_INTERVAL
VERBOSITY = True

result_data = {
    'results': {},
    'meta': {}
}

data = retrieve_data(FLOW_INTERVAL)

# moving_average(data)

# naive(data)

# random_forest(data, False)

# random_forest_grid(data, False)

# random_forest_grid(data, True)

# support_vector_machine(data, False)

# support_vector_machine_grid(data, False)

support_vector_machine_grid(data, True)

# lstm(data, False)

# lstm_grid(data, False)

# lstm_grid(data, True)

# gru(data, False)

# gru_grid(data, False)

# gru_grid(data, True)

"""## Compare"""

# VERBOSITY = False

# predict_futures = [15, 30, 45, 60]
# compare_results_by_predict_in_future(predict_futures)

# flow_intervals = [150, 300, 450]
# compare_results_by_flow_interval(flow_intervals)

# seeable_pasts = [60, 120, 240, 480]
# compare_results_by_seeable_past(seeable_pasts)

# n_splits = [1, 2, 4, 8]
# compare_results_by_n_split(n_splits)

"""## Observations:

+ For the evaluation of the RNN and it's variations was used the Walking Forward methodology so that we had many test sessions and all training sessions where the same size [[1]](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9)
+ To remove the cross-validation of the GridSearchCV we based on the answer in [scikit learn discussion - allow GridSearchCV to work with params={} or cv=1](https://github.com/scikit-learn/scikit-learn/issues/2048)
+ Grid Search on Keras was based on the article [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
"""