# methods/catalog/care/trained_models/train_models.py

"""
Train and save model for CARE benchmark experiments.

Usage:
    cd methods/catalog/care/trained_models
    python train_models.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add paths
CURRENT_FILE = os.path.abspath(__file__)
TRAINED_MODELS_DIR = os.path.dirname(CURRENT_FILE)
CARE_DIR = os.path.dirname(TRAINED_MODELS_DIR)
LIBRARY_DIR = os.path.join(CARE_DIR, 'library')
sys.path.insert(0, LIBRARY_DIR)

from prepare_datasets import PrepareAdult  # type: ignore
from create_model import CreateModel  # type: ignore


def main():
    """Train and save model"""
    dataset_path = os.path.join(CARE_DIR, 'datasets/')
    
    print('Loading Adult dataset...')
    dataset = PrepareAdult(dataset_path, 'adult.csv')
    
    # Split data
    X, y = dataset['X_ord'], dataset['y']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print('Training Gradient Boosting classifier...')
    blackbox = CreateModel(
        dataset, X_train, X_test, Y_train, Y_test, 
        'classification', 'gb-c', GradientBoostingClassifier
    )
    
    # Save model
    print('Saving model...')
    model_path = os.path.join(TRAINED_MODELS_DIR, 'adult_gb_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(blackbox, f)
    
    print(f'Model saved to: {model_path}')
    


if __name__ == '__main__':
    main()