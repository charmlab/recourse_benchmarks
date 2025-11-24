import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import io
import contextlib

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Add paths
CURRENT_FILE = os.path.abspath(__file__)
CARE_DIR = os.path.dirname(CURRENT_FILE)
LIBRARY_DIR = os.path.join(CARE_DIR, 'library')
sys.path.insert(0, LIBRARY_DIR)

from prepare_datasets import PrepareAdult  # type: ignore
from create_model import CreateModel  # type: ignore
from user_preferences import userPreferences  # type: ignore
from care.care import CARE  # type: ignore
from evaluate_counterfactuals import evaluateCounterfactuals  # type: ignore


def reproduce_care_table7(n_samples=50, n_cf=10, verbose=True):
    """
    Reproduce CARE Table 7 results for Adult dataset.
    """
    dataset_path = os.path.join(CARE_DIR, 'datasets/')
    
    if verbose:
        print('=' * 60)
        print('CARE Table 7 Reproduction')
        print('=' * 60)
    
    # Load Adult dataset (suppress print statements)
    if verbose:
        print('\n[1/5] Loading Adult dataset...')
    dataset = PrepareAdult(dataset_path, 'adult.csv')
    
    # Load pre-trained model
    if verbose:
        print('[2/5] Loading pre-trained model...')
    
    model_path = os.path.join(CARE_DIR, 'trained_models', 'adult_gb_classifier.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model not found at: {model_path}\n"
            f"Please run: cd trained_models && python train_models.py"
        )
    
    with open(model_path, 'rb') as f:
        blackbox = pickle.load(f)
    
    # Split data
    X, y = dataset['X_ord'], dataset['y']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    predict_fn = lambda x: blackbox.predict(x).ravel()
    predict_proba_fn = lambda x: blackbox.predict_proba(x)
    
    # Create CARE configurations (suppress prints)
    if verbose:
        print('[3/5] Initializing CARE configurations...')
    
    configs = []
    config_names = ['{1}', '{1,2}', '{1,2,3}', '{1,2,3,4}']
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        for i, (soundness, coherency, actionability) in enumerate([
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True)
        ]):
            care = CARE(
                dataset, task='classification',
                predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                SOUNDNESS=soundness, COHERENCY=coherency, ACTIONABILITY=actionability,
                n_cf=n_cf
            )
            care.fit(X_train, Y_train)
            configs.append(care)
    
    # Store results
    task = 'classification'
    n_out = int(task == 'classification') + 1
    results_all = {name: [] for name in config_names}
    
    # Explain instances
    if verbose:
        print(f'[4/5] Generating explanations for {n_samples} instances...')
    
    explained = 0
    for idx, x_ord in enumerate(X_test):
        try:
            # Suppress all prints during explanation generation
            with contextlib.redirect_stdout(f):
                # Generate explanations for all configs
                explanations = []
                for i, care_model in enumerate(configs):
                    if i == 3:  # Config {1,2,3,4} needs user preferences
                        user_prefs = userPreferences(dataset, x_ord)
                        exp = care_model.explain(x_ord, user_preferences=user_prefs)
                    else:
                        exp = care_model.explain(x_ord)
                    explanations.append(exp)
                
                # Use last config's toolbox for evaluation
                toolbox = explanations[-1]['toolbox']
                objective_names = explanations[-1]['objective_names']
                featureScaler = explanations[-1]['featureScaler']
                feature_names = dataset['feature_names']
                
                # Evaluate each config
                for config_name, explanation in zip(config_names, explanations):
                    _, _, x_cfs_ord, x_cfs_eval = evaluateCounterfactuals(
                        x_ord, explanation['cfs_ord'], dataset,
                        predict_fn, predict_proba_fn, task, toolbox,
                        objective_names, featureScaler, feature_names
                    )
                    
                    # Find best CF
                    idx_best = (np.where((x_cfs_ord == 
                                         explanation['best_cf_ord']).all(axis=1)==True))[0][0]
                    
                    # Store metrics
                    results_all[config_name].append(
                        x_cfs_eval.iloc[idx_best, :-n_out].values
                    )
            
            explained += 1
            
            # Print progress every 10 instances
            if verbose and explained % 10 == 0:
                print(f'  Progress: {explained}/{n_samples}')
                
        except Exception as e:
            if verbose:
                print(f'  Warning: Failed to explain instance {idx}: {e}')
        
        if explained == n_samples:
            break
    
    # Compute statistics
    if verbose:
        print('[5/5] Computing statistics...')
    
    metric_names = ['outcome', 'actionability', 'coherency', 'proximity', 
                   'connectedness', 'distance', 'sparsity']
    
    results = {}
    for config_name, config_results in results_all.items():
        if len(config_results) > 0:
            results_array = np.array(config_results)
            means = np.mean(results_array, axis=0)
            stds = np.std(results_array, axis=0)
            
            results[config_name] = {}
            for i, metric in enumerate(metric_names):
                results[config_name][metric] = {
                    'mean': means[i],
                    'std': stds[i]
                }
    
    return results


def print_table7_format(results):
    """Print results in Table 7 format"""
    print('\n' + '=' * 120)
    print('RESULTS (Table 7 Format)')
    print('=' * 120)
    
    # Header
    print(f"{'Config':<15} {'↓O_outcome':<15} {'↓O_distance':<15} {'↓O_sparsity':<15} "
          f"{'↑O_proximity':<15} {'↑O_connect':<15} {'↓O_cohere':<15} {'↓O_action':<15}")
    print('-' * 120)
    
    # Rows
    for config in ['{1}', '{1,2}', '{1,2,3}', '{1,2,3,4}']:
        if config in results:
            r = results[config]
            print(f"{config:<15} "
                  f"{r['outcome']['mean']:.2f}±{r['outcome']['std']:.1f}      "
                  f"{r['distance']['mean']:.2f}±{r['distance']['std']:.1f}      "
                  f"{r['sparsity']['mean']:.2f}±{r['sparsity']['std']:.1f}      "
                  f"{r['proximity']['mean']:.2f}±{r['proximity']['std']:.1f}      "
                  f"{r['connectedness']['mean']:.2f}±{r['connectedness']['std']:.1f}      "
                  f"{r['coherency']['mean']:.2f}±{r['coherency']['std']:.1f}      "
                  f"{r['actionability']['mean']:.2f}±{r['actionability']['std']:.1f}")
    print('=' * 120)


def assert_table7_values(results, tolerance=0.15):
    """Assert results match Table 7 expected values"""
    expected = {
        '{1}': {
            'outcome': 0.00, 'distance': 0.02, 'sparsity': 1.41,
            'proximity': 0.58, 'connectedness': 0.20, 'coherency': 0.05, 'actionability': 0.08
        },
        '{1,2}': {
            'outcome': 0.00, 'distance': 0.12, 'sparsity': 2.84,
            'proximity': 1.00, 'connectedness': 1.00, 'coherency': 0.10, 'actionability': 0.36
        },
        '{1,2,3}': {
            'outcome': 0.00, 'distance': 0.12, 'sparsity': 3.05,
            'proximity': 1.00, 'connectedness': 1.00, 'coherency': 0.00, 'actionability': 0.35
        },
        '{1,2,3,4}': {
            'outcome': 0.00, 'distance': 0.11, 'sparsity': 2.76,
            'proximity': 1.00, 'connectedness': 0.91, 'coherency': 0.00, 'actionability': 0.00
        }
    }
    
    print('\n' + '=' * 80)
    print('ASSERTIONS')
    print('=' * 80)
    
    all_passed = True
    for config in ['{1}', '{1,2}', '{1,2,3}', '{1,2,3,4}']:
        print(f'\nConfig {config}:')
        for metric, expected_val in expected[config].items():
            actual_val = results[config][metric]['mean']
            diff = abs(actual_val - expected_val)
            
            # Use relative or absolute tolerance
            if expected_val > 0.01:
                passed = (diff / expected_val) <= tolerance
            else:
                passed = diff <= 0.05
            
            status = '✓' if passed else '✗'
            print(f'  {status} {metric:15s}: {actual_val:.3f} (expected {expected_val:.2f}, diff {diff:.3f})')
            
            if not passed:
                all_passed = False
    
    print('\n' + '=' * 80)
    if all_passed:
        print('✓ ALL ASSERTIONS PASSED')
    else:
        print('✗ SOME ASSERTIONS FAILED')
    print('=' * 80)
    
    return all_passed


def main():
    """Main function"""
    # Run with 50 samples for testing (change to 500 for full reproduction)
    results = reproduce_care_table7(n_samples=50, n_cf=10, verbose=True)
    
    # Print results
    print_table7_format(results)
    
    # Run assertions (use higher tolerance for small sample size)
    assert_table7_values(results, tolerance=0.15)
    
    return results


if __name__ == '__main__':
    results = main()