import pytest
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from recourse_methods import Greedy
from models.negative_instances import predict_negative_instances

"""
The tests focus on two key aspects:
1. Minimal perturbations (ensuring small changes between factuals and counterfactuals).
2. Realism (ensuring counterfactuals belong to the data distribution of the target class).

Minimal Perturbations using L1 Distance
The research paper emphasizes that counterfactual explanations should have minimal perturbations, meaning the smallest necessary changes are made to the input (factual) to change the classification outcome.

Realism using IM1 Score
The paper introduces the IM1 score as a measure of realism for the generated counterfactuals. A counterfactual is considered realistic if it resembles data points from the target class distribution.

Implented from:
"Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties"
Lisa Schut, Oscar Key, Rory McGrathz, Luca Costabelloz, Bogdan Sacaleanuz, Medb Corcoranz, Yarin Galy.
"""

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def calculate_standard_deviation(values):
    return np.std(values)

@pytest.mark.parametrize("dataset_name, model_type, backend", [
    ("breast_cancer", "linear", "tensorflow"),
    ("boston_housing", "linear", "tensorflow")
])

def test_greedy_on_datasets(dataset_name, model_type, backend):
    data = DataCatalog(dataset_name, model_type, 0.7)
    model = ModelCatalog(data, model_type, backend=backend)
    
    greedy = Greedy(mlmodel=model)
    
    total_factuals = predict_negative_instances(model, data)
    
    factuals = total_factuals.iloc[:5]
    df = data.df
    
    df2 = df.drop(columns=['y'])
    input_dim = df2.shape[1]
    ae_all = build_autoencoder(input_dim)
    ae_all.fit(df2, df2, epochs=50, batch_size=32, verbose=0)
    
    target_class_data = df[df['y'] == 1].drop(columns=['y'])
    ae_target = build_autoencoder(input_dim)
    ae_target.fit(target_class_data, target_class_data, epochs=50, batch_size=32, verbose=0)
    
    l1_distances = []
    realism_scores = []
    
    counterfactuals = greedy.get_counterfactuals(factuals)
        
    l1_distance = calculate_l1_distance(counterfactuals, factuals)
    realism_score = calculate_realism_score(counterfactuals, ae_all, ae_target)
        
    l1_distances.append(l1_distance)
    realism_scores.append(realism_score)
    
    l1_std = calculate_standard_deviation(l1_distances)
    realism_std = calculate_standard_deviation(realism_scores)
    
    assert l1_std < 0.1
    assert realism_std < 0.1

def calculate_l1_distance(counterfactuals, factuals):
    return np.mean(np.abs(counterfactuals - factuals).sum(axis=1))

def calculate_realism_score(counterfactuals, ae_all, ae_target):
    if isinstance(counterfactuals, pd.DataFrame):
        counterfactuals = counterfactuals.values

    reconstruction_all = ae_all.predict(counterfactuals)
    loss_all = np.sum(np.square(counterfactuals - reconstruction_all), axis=1)

    reconstruction_target = ae_target.predict(counterfactuals)
    loss_target = np.sum(np.square(counterfactuals - reconstruction_target), axis=1)

    epsilon = 1e-8
    im1_scores = loss_target / (loss_all + epsilon)
    
    return np.mean(im1_scores)
