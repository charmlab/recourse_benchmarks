def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn # to ignore all warnings.

import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
import loadData

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import torch
from tensorflow import keras
from tensorflow.keras.layers import Dense

import xgboost as xgb

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

# TODO: change to be like _data_main below, and make python module
# this answer https://stackoverflow.com/a/50474562 and others
try:
  import treeUtils
except:
  print('[ENV WARNING] treeUtils not available')

SIMPLIFY_TREES = False

# Custom module for Neural Networks
class PyTorchNeuralNetwork(torch.nn.Module):    
  # Constructor
  def __init__(self, n_inputs, n_outputs, n_neurons):
    super(PyTorchNeuralNetwork, self).__init__()
    self.fc1 = torch.nn.Linear(n_inputs, n_neurons)
    self.fc2 = torch.nn.Linear(n_neurons, n_neurons)
    self.fc3 = torch.nn.Linear(n_neurons, n_outputs)
  # Predictions
  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    y_pred = torch.nn.functional.softmax(self.fc3(x))
    return y_pred
  
  def fit(self, x_train, y_train):
    x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
    y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True) 
  
    # defining the optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # defining Cross-Entropy loss
    criterion = torch.nn.NLLLoss()
    
    epochs = 1
    for _ in range(epochs):
      for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = self(data)
        output = torch.log(output)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    return self

  def predict(self, test):
    self.eval()
    y_train_pred = []
    with torch.no_grad():
      output = self(test)

      y_train_pred.extend(output)
    
    y_train_pred = torch.stack(y_train_pred)
    return y_train_pred

# Custom module for logistic regression
class PyTorchLogisticRegression(torch.nn.Module):    
  # Constructor
  def __init__(self, n_inputs, n_outputs):
    super(PyTorchLogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(n_inputs, n_outputs)

  # Predictions
  def forward(self, x):
    y_pred = torch.nn.functional.softmax(self.linear(x))
    return y_pred
  
  def fit(self, x_train, y_train):
    x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32))
    y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True) 
  
    # Defining the optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # Defining Cross-Entropy loss
    criterion = torch.nn.NLLLoss()
    
    epochs = 1
    for _ in range(epochs):
      for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = self(data)
        output = torch.log(output)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    return self

  def predict(self, test):
    self.eval()
    y_train_pred = []
    with torch.no_grad():
      output = self(test)

      y_train_pred.extend(output)
    
    y_train_pred = torch.stack(y_train_pred)
    return y_train_pred

class TenserflowNeuralNetwork(keras.Model):
    def __init__(self, n_inputs, n_outputs, n_neurons):
      super(TenserflowNeuralNetwork, self).__init__()
      self.fc1 = Dense(n_neurons, activation="relu", input_dim=n_inputs)
      self.fc2 = Dense(n_neurons, activation="relu", input_dim=n_neurons)
      self.fc3 = Dense(n_outputs, activation="softmax", input_dim=n_neurons)

    def call(self, inputs, training=False):
      fc1_out = self.fc1(inputs)
      fc2_out = self.fc2(fc1_out)
      y_pred = self.fc3(fc2_out)
      return y_pred
    
    def fit(self, x_train, y_train):
      self.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
      super(TenserflowNeuralNetwork, self).fit(x_train, y_train)

      return self
    
    def predict(self, test):
      # Predict method
      output = super(TenserflowNeuralNetwork, self).predict(test)
      return output

class TenserflowLogisticRegression(keras.Model):
    def __init__(self, n_inputs, n_outputs):
      super(TenserflowLogisticRegression, self).__init__()
      self.linear = Dense(n_outputs, activation="softmax", input_dim=n_inputs)

    def call(self, inputs, training=False):
      y_pred = self.linear(inputs)
      return y_pred
    
    def fit(self, x_train, y_train):
      self.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
      super(TenserflowLogisticRegression, self).fit(x_train, y_train)

      return self
    
    def predict(self, test):
      # Predict method
      output = super(TenserflowLogisticRegression, self).predict(test)
      return output
    
@utils.Memoize
def loadModelForDataset(model_class, dataset_string, backend='sklearn', scm_class = None, experiment_folder_name = None):
  """
  Loads and returns a model with trained data.

  Parameters
  ----------
  model_class : str
    Class of model to be trained.
  dataset_string : str
      Dataset to train model with.
  backend : str
      The backend of the model.
  scm_class : object
      SCM Class of the retrieved dataset.
  experiment_folder_name : str
      Folder name to save model in.
      
  Returns
  -------
  model :  The trained model.
  """
  log_file = sys.stdout if experiment_folder_name == None else open(f'{experiment_folder_name}/log_training.txt','w')

  if not (model_class in {'linear', 'mlp', 'tree', 'forest'}):
    raise Exception(f'{model_class} not supported.')

  if not (dataset_string in {'synthetic', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult', 'test'}):
    raise Exception(f'{dataset_string} not supported.')

  if model_class in {'tree', 'forest'}:
    one_hot = False
  elif model_class in { 'mlp', 'linear'}:
    one_hot = True
  else:
    raise Exception(f'{model_class} not recognized as a valid `model_class`.')

  dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = True, meta_param = scm_class)
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit(preprocessing = 'normalize')
  X_all = pd.concat([X_train, X_test], axis = 0)
  y_all = pd.concat([y_train, y_test], axis = 0)
  assert sum(y_all) / len(y_all) == 0.5, 'Expected class balance should be 50/50%.'
  feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)

  logisticRegressionMap = {
    'sklearn': LogisticRegression(solver='liblinear'),     # IMPORTANT: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22; therefore, results may differ slightly from paper.
    'pytorch': PyTorchLogisticRegression(X_train.shape[1], 2),
    'tensorflow': TenserflowLogisticRegression(X_train.shape[1], 2)
  }

  neuralNetworksMap = {
    'sklearn': MLPClassifier(hidden_layer_sizes = (10, 10)),
    'pytorch': PyTorchNeuralNetwork(X_train.shape[1], 2, 10),
    'tensorflow': TenserflowNeuralNetwork(X_train.shape[1], 2, 10)
  }

  forestMap = {
    'sklearn': RandomForestClassifier(),
    'xgboost': xgb.XGBClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2, tree_method="hist", early_stopping_rounds=2)
  }

  if model_class == 'tree':
    model_pretrain = DecisionTreeClassifier()
  elif model_class == 'forest':
    model_pretrain = forestMap[backend]
  elif model_class == 'linear':
    model_pretrain = logisticRegressionMap[backend]
  elif model_class == 'mlp':
    model_pretrain = neuralNetworksMap[backend]

  tmp_text = f'[INFO] Training `{model_class}` on {X_train.shape[0]:,} samples ' + \
    f'(%{100 * X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f} ' + \
    f'of {X_train.shape[0] + X_test.shape[0]:,} samples)...'
  print(tmp_text)
  print(tmp_text, file=log_file)

  model_trained = model_pretrain.fit(X_train.values, y_train.values)
  classifier_obj = model_trained
  # visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name)

  if model_class == 'tree':
    if SIMPLIFY_TREES:
      print('[INFO] Simplifying decision tree...', end = '', file=log_file)
      model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
      print('\tdone.', file=log_file)
    # treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
  elif model_class == 'forest':
    for tree_idx in range(len(model_trained.estimators_)):
      if SIMPLIFY_TREES:
        print(f'[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...', end = '', file=log_file)
        model_trained.estimators_[tree_idx].tree_ = treeUtils.simplifyDecisionTree(model_trained.estimators_[tree_idx], False)
        print('\tdone.', file=log_file)
      # treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)

  if experiment_folder_name:
    pickle.dump(model_trained, open(f'{experiment_folder_name}/_model_trained', 'wb'))

  return model_trained


def scatterDataset(dataset_obj, classifier_obj, ax):
  assert len(dataset_obj.getInputAttributeNames()) <= 3
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  number_of_samples_to_plot = min(200, X_train_numpy.shape[0], X_test_numpy.shape[0])
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train[idx] == 1 else 'magenta'
    color_test = 'black' if y_test[idx] == 1 else 'magenta'
    if X_train.shape[1] == 2:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], marker='o', color=color_test, alpha=0.2, s=15)
    elif X_train.shape[1] == 3:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterDecisionBoundary(dataset_obj, classifier_obj, ax):

  if len(dataset_obj.getInputAttributeNames()) == 2:

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 1000)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 1000)
    X, Y = np.meshgrid(X, Y)
    Xp = X.ravel()
    Yp = Y.ravel()

    # if normalized_fixed_model is False:
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    # else:
    #   Xp = (Xp - dataset_obj.attributes_kurz['x0'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x0'].upper_bound - dataset_obj.attributes_kurz['x0'].lower_bound)
    #   Yp = (Yp - dataset_obj.attributes_kurz['x1'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x1'].upper_bound - dataset_obj.attributes_kurz['x1'].lower_bound)
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    labels = classifier_obj.predict(np.c_[Xp, Yp])
    Z = labels.reshape(X.shape)

    cmap = plt.get_cmap('Paired')
    ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)

  elif len(dataset_obj.getInputAttributeNames()) == 3:

    fixed_model_w = classifier_obj.coef_
    fixed_model_b = classifier_obj.intercept_

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
    X, Y = np.meshgrid(X, Y)
    Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

    surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)



def visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name):

  if not len(dataset_obj.getInputAttributeNames()) <= 3:
    return

  fig = plt.figure()
  if len(dataset_obj.getInputAttributeNames()) == 2:
    ax = plt.subplot()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid()
  elif len(dataset_obj.getInputAttributeNames()) == 3:
    ax = plt.subplot(1, 1, 1, projection = '3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.view_init(elev=10, azim=-20)

  scatterDataset(dataset_obj, classifier_obj, ax)
  scatterDecisionBoundary(dataset_obj, classifier_obj, ax)

  ax.set_title(f'{dataset_obj.dataset_name}')
  ax.grid(True)

  # plt.show()
  plt.savefig(f'{experiment_folder_name}/_dataset_and_model.pdf')
  plt.close()