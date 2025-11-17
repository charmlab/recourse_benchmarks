

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dataset(ABC):
    def __init__(self, seed: int = 0, n_folds: int = 5):
        super().__init__()
        self.seed = seed
        self.n_folds = n_folds
        self.X = None
        self.y = None
        self.X_shift = None
        self.y_shift = None
        self.name = None
        self.scaler = None
    
    def get_feature_types(self, df: pd.DataFrame):
        cat_features, num_features = [], []
        for feature in df.columns:
            if df[feature].dtype == object:
                cat_features.append(feature)
            elif len(set(df[feature])) > 2:
                num_features.append(feature)
        return cat_features, num_features
    
    def scale_num_features(self, df: pd.DataFrame, num_features: List[str]):
        self.scaler = StandardScaler()
        df[num_features] = self.scaler.fit_transform(df[num_features].values)
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, fold: int):
        X_chunks, y_chunks = [], []
        for i in range(self.n_folds):
            start = int(i/self.n_folds * len(X))
            end = int((i+1)/self.n_folds * len(X))
            X_chunks.append(X.iloc[start:end])
            y_chunks.append(y.iloc[start:end])
            
        X_test, y_test = X_chunks.pop(fold), y_chunks.pop(fold)
        X_train, y_train = pd.concat(X_chunks), pd.concat(y_chunks)
        
        return (X_train, y_train), (X_test, y_test)
    
    def get_data(self, fold: int, shift: bool = False):
        fold = fold % self.n_folds
        if shift:
            return self.split_data(self.X, self.y, fold), self.split_data(self.X_shift, self.y_shift, fold)
        else:
            return self.split_data(self.X, self.y, fold)

class GermanDataset(Dataset):
    def __init__(self, seed = 0, n_folds = 5):
        super(GermanDataset, self).__init__(seed, n_folds)
        self.X, self.y = self.create('methods/catalog/larr/german.csv')
        self.X_shift, self.y_shift = self.create('methods/catalog/larr/corrected_german.csv')
        self.name = 'german'
        self.cat_features, self.num_features = list(range(3,7)), list(range(2))
        self.imm_features = [2]
        
    def create(self, filepath):
        df = pd.read_csv(filepath, sep=',').sample(frac=1, random_state=self.seed)
        
        cat_features, num_features = ['personal_status_sex'], ['duration', 'amount', 'age']
        
        target = 'credit_risk'
        
        df = df.drop(columns=[c for c in list(df) if c not in num_features+cat_features+[target]])
        df = self.scale_num_features(df, num_features)
        df = pd.get_dummies(df, columns=cat_features, dtype=float)
        
        X, y = df.drop(columns=[target]), df[target]
        return X, y
    
class Model(ABC):
	def __init__(self):
		pass

	def metrics(self, X, y):
		acc = np.mean(self.predict(X)==y)

		pred = self.predict_proba(X)[:,1]
		fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)

		return acc, auc
 
	@abstractmethod
	def train(self, X, y):
		pass

class NN(Model):
	def __init__(self, n_features):
		torch.manual_seed(0)
		super(NN, self).__init__()
		self.model = nn.Sequential(
		  nn.Linear(n_features, 50),
		  nn.ReLU(),
		  nn.Linear(50, 100),
		  nn.ReLU(),
		  nn.Linear(100, 200),
		  nn.ReLU(),
		  nn.Linear(200, 1),
		  nn.Sigmoid()
		  )
	
	def torch_model(self,x):
		return self.model(x)[0]

	def train(self, X_train, y_train, verbose=0):
		torch.manual_seed(0)
		X_train = torch.from_numpy(X_train).float()
		y_train = torch.from_numpy(y_train).float()

		loss_fn = nn.BCELoss()
		optimizer = torch.optim.Adam(self.model.parameters())

		epochs = 100
		for epoch in range(epochs):
			self.model.train()
			optimizer.zero_grad()

			y_pred = self.model(X_train)
			loss = loss_fn(y_pred[:,0], y_train)
			if verbose: print(f'Epoch {epoch}: train loss: {loss.item()}')

			loss.backward()
			optimizer.step()

	def predict_proba(self, X: np.ndarray):
		X = torch.from_numpy(np.array(X)).float()
		class1_probs = self.model(X).detach().numpy()
		class0_probs = 1-class1_probs
		return np.hstack((class0_probs,class1_probs))

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)