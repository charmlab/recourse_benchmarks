"""
Reproduce CARE Table 7 results.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Add library to path (same as model.py)
CURRENT_FILE = os.path.abspath(__file__)
CARE_DIR = os.path.dirname(CURRENT_FILE)
LIBRARY_DIR = os.path.join(CARE_DIR, 'library')
sys.path.insert(0, LIBRARY_DIR)

# Direct imports from library/
from prepare_datasets import PrepareAdult  # type: ignore
from create_model import CreateModel  # type: ignore
from user_preferences import userPreferences  # type: ignore
from care.care import CARE  # type: ignore
from evaluate_counterfactuals import evaluateCounterfactuals  # type: ignore


def reproduce_care_table7(n_samples=50, n_cf=10):
    """
    Reproduce CARE Table 7 results for Adult dataset.
    
    Args:
        n_samples: Number of instances to explain (default: 50, paper uses 500)
        n_cf: Number of counterfactuals per instance (default: 10)
    
    Returns:
        dict: Results for all configurations
    """
    # Define paths
    dataset_path = os.path.join(CARE_DIR, 'datasets/')
    
    print('=' * 60)
    print('CARE Table 7 Reproduction')
    print('=' * 60)
    
    # Load Adult dataset
    print('\nLoading Adult dataset...')
    dataset_name = 'adult.csv'
    dataset = PrepareAdult(dataset_path, dataset_name)
    
    # Split data
    X, y = dataset['X_ord'], dataset['y']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print('Training Gradient Boosting classifier...')
    task = 'classification'
    blackbox_name = 'gb-c'
    blackbox = CreateModel(
        dataset, X_train, X_test, Y_train, Y_test, 
        task, blackbox_name, GradientBoostingClassifier
    )
    
    predict_fn = lambda x: blackbox.predict(x).ravel()
    predict_proba_fn = lambda x: blackbox.predict_proba(x)
    
    # Create CARE configurations
    print('Initializing CARE configurations...')
    care_config_1 = CARE(
        dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
        SOUNDNESS=False, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf
    )
    care_config_1.fit(X_train, Y_train)
    
    care_config_12 = CARE(
        dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
        SOUNDNESS=True, COHERENCY=False, ACTIONABILITY=False, n_cf=n_cf
    )
    care_config_12.fit(X_train, Y_train)
    
    care_config_123 = CARE(
        dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
        SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=False, n_cf=n_cf
    )
    care_config_123.fit(X_train, Y_train)
    
    care_config_1234 = CARE(
        dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
        SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True, n_cf=n_cf
    )
    care_config_1234.fit(X_train, Y_train)
    
    # Store results
    n_out = int(task == 'classification') + 1
    results_config_1 = []
    results_config_12 = []
    results_config_123 = []
    results_config_1234 = []
    
    # Explain instances
    print(f'Generating explanations for {n_samples} instances...')
    explained = 0
    
    for x_ord in X_test:
        try:
            # Generate explanations
            explanation_config_1 = care_config_1.explain(x_ord)
            explanation_config_12 = care_config_12.explain(x_ord)
            explanation_config_123 = care_config_123.explain(x_ord)
            user_preferences = userPreferences(dataset, x_ord)
            explanation_config_1234 = care_config_1234.explain(
                x_ord, user_preferences=user_preferences
            )
            
            # Get toolbox for evaluation (from last config)
            toolbox = explanation_config_1234['toolbox']
            objective_names = explanation_config_1234['objective_names']
            featureScaler = explanation_config_1234['featureScaler']
            feature_names = dataset['feature_names']
            
            # Evaluate config 1
            cfs_ord_config_1, cfs_eval_config_1, x_cfs_ord_config_1, x_cfs_eval_config_1 = \
                evaluateCounterfactuals(
                    x_ord, explanation_config_1['cfs_ord'], dataset,
                    predict_fn, predict_proba_fn, task, toolbox,
                    objective_names, featureScaler, feature_names
                )
            idx_best_config_1 = (np.where((x_cfs_ord_config_1 == 
                                          explanation_config_1['best_cf_ord']).all(axis=1)==True))[0][0]
            results_config_1.append(x_cfs_eval_config_1.iloc[idx_best_config_1, :-n_out].values)
            
            # Evaluate config 12
            cfs_ord_config_12, cfs_eval_config_12, x_cfs_ord_config_12, x_cfs_eval_config_12 = \
                evaluateCounterfactuals(
                    x_ord, explanation_config_12['cfs_ord'], dataset,
                    predict_fn, predict_proba_fn, task, toolbox,
                    objective_names, featureScaler, feature_names
                )
            idx_best_config_12 = (np.where((x_cfs_ord_config_12 == 
                                           explanation_config_12['best_cf_ord']).all(axis=1)==True))[0][0]
            results_config_12.append(x_cfs_eval_config_12.iloc[idx_best_config_12, :-n_out].values)
            
            # Evaluate config 123
            cfs_ord_config_123, cfs_eval_config_123, x_cfs_ord_config_123, x_cfs_eval_config_123 = \
                evaluateCounterfactuals(
                    x_ord, explanation_config_123['cfs_ord'], dataset,
                    predict_fn, predict_proba_fn, task, toolbox,
                    objective_names, featureScaler, feature_names
                )
            idx_best_config_123 = (np.where((x_cfs_ord_config_123 == 
                                            explanation_config_123['best_cf_ord']).all(axis=1)==True))[0][0]
            results_config_123.append(x_cfs_eval_config_123.iloc[idx_best_config_123, :-n_out].values)
            
            # Evaluate config 1234
            cfs_ord_config_1234, cfs_eval_config_1234, x_cfs_ord_config_1234, x_cfs_eval_config_1234 = \
                evaluateCounterfactuals(
                    x_ord, explanation_config_1234['cfs_ord'], dataset,
                    predict_fn, predict_proba_fn, task, toolbox,
                    objective_names, featureScaler, feature_names
                )
            idx_best_config_1234 = (np.where((x_cfs_ord_config_1234 == 
                                             explanation_config_1234['best_cf_ord']).all(axis=1)==True))[0][0]
            results_config_1234.append(x_cfs_eval_config_1234.iloc[idx_best_config_1234, :-n_out].values)
            
            explained += 1
            if explained % 10 == 0:
                print(f'  Progress: {explained}/{n_samples}')
                
        except Exception as e:
            print(f'  Warning: Failed to explain instance: {e}')
            pass
        
        if explained == n_samples:
            break
    
    # Compute statistics
    print('\nComputing statistics...')
    results = {}
    
    # Metric names from evaluateCounterfactuals
    metric_names = ['outcome', 'actionability', 'coherency', 'proximity', 
                   'connectedness', 'distance', 'sparsity']
    
    for config_name, config_results in [
        ('{1}', results_config_1),
        ('{1,2}', results_config_12),
        ('{1,2,3}', results_config_123),
        ('{1,2,3,4}', results_config_1234)
    ]:
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
    print('\n' + '=' * 100)
    print('RESULTS (Table 7 Format)')
    print('=' * 100)
    
    # Header
    print(f"{'Config':<15} {'↓O_outcome':<15} {'↓O_distance':<15} {'↓O_sparsity':<15} "
          f"{'↑O_proximity':<15} {'↑O_connectedness':<15} {'↓O_coherency':<15} {'↓O_actionability':<15}")
    print('-' * 100)
    
    # Rows
    for config in ['{1}', '{1,2}', '{1,2,3}', '{1,2,3,4}']:
        if config in results:
            r = results[config]
            print(f"{config:<15} "
                  f"{r['outcome']['mean']:.2f}±{r['outcome']['std']:.1f}  "
                  f"{r['distance']['mean']:.2f}±{r['distance']['std']:.1f}  "
                  f"{r['sparsity']['mean']:.2f}±{r['sparsity']['std']:.1f}  "
                  f"{r['proximity']['mean']:.2f}±{r['proximity']['std']:.1f}  "
                  f"{r['connectedness']['mean']:.2f}±{r['connectedness']['std']:.1f}  "
                  f"{r['coherency']['mean']:.2f}±{r['coherency']['std']:.1f}  "
                  f"{r['actionability']['mean']:.2f}±{r['actionability']['std']:.1f}")
    print('=' * 100)


def assert_table7_values(results, tolerance=0.10):
    """Assert results match Table 7 expected values"""
    expected = {
        '{1}': {
            'outcome': 0.00, 'distance': 0.02, 'sparsity': 1.41,
            'proximity': 0.58, 'connectedness': 0.20, 'coherency': 0.05, 
            'actionability': 0.08
        },
        '{1,2}': {
            'outcome': 0.00, 'distance': 0.12, 'sparsity': 2.84,
            'proximity': 1.00, 'connectedness': 1.00, 'coherency': 0.10, 
            'actionability': 0.36
        },
        '{1,2,3}': {
            'outcome': 0.00, 'distance': 0.12, 'sparsity': 3.05,
            'proximity': 1.00, 'connectedness': 1.00, 'coherency': 0.00, 
            'actionability': 0.35
        },
        '{1,2,3,4}': {
            'outcome': 0.00, 'distance': 0.11, 'sparsity': 2.76,
            'proximity': 1.00, 'connectedness': 0.91, 'coherency': 0.00, 
            'actionability': 0.00
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
            print(f'  {status} {metric:15s}: {actual_val:.3f} '
                  f'(expected {expected_val:.2f}, diff {diff:.3f})')
            
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
    results = reproduce_care_table7(n_samples=50, n_cf=10)
    
    # Print results
    print_table7_format(results)
    
    # Run assertions (use higher tolerance for small sample size)
    assert_table7_values(results, tolerance=0.15)
    
    return results


if __name__ == '__main__':
    results = main()