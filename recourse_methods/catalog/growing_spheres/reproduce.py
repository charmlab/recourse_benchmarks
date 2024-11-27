from random import seed
import pytest
import numpy as np
import pandas as pd

from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from recourse_methods import GrowingSpheres
from models.negative_instances import predict_negative_instances

RANDOM_SEED = 54321
seed(
    RANDOM_SEED
)  # set the random seed so that the random permutations can be reproduced again

"""
The test is designed to replicate the sparsity results described in the research paper.
By comparing the calculated maximum number of features changed for generated counterfactuals
with the expected range, the test ensures that the generated counterfactuals are consistent
with paper's findings.

Implemented from:
"Laugel, T., Lesot, M. J., Marsala, C., Renard, X., & Detyniecki, M. (2017). 
Inverse classification for comparison-based interpretability in machine learning. arXiv preprint arXiv:1712.08443.
"""



@pytest.mark.parametrize("dataset_name", [
    ("online_news_popularity")
])

def test_growing_spheres_sparsity_news(dataset_name):
        """
        Test the sparsity of explanations for the news popularity dataset
        """
        data = DataCatalog(dataset_name, "forest", 0.7)
        model = ModelCatalog(data, "forest", backend="sklearn")

        gs = GrowingSpheres(mlmodel=model)
        total_factuals = predict_negative_instances(model, data)
        factuals = total_factuals.sample(n=100, random_state=RANDOM_SEED)

        counterfactuals = gs.get_counterfactuals(factuals)
        
        mask = ~counterfactuals.isnull().any(axis=1)  # Mask for rows without NaN in counterfactuals

        # Apply the mask to both DataFrames
        aligned_factuals = factuals[mask].reset_index(drop=True).iloc[:, :-1]
        aligned_counterfactuals = counterfactuals[mask].reset_index(drop=True)

        explanation_vectors = aligned_factuals.values - aligned_counterfactuals.values

        # Compute the L0 norm (count of non-zero elements) 
        sparsity_scores = np.count_nonzero(explanation_vectors, axis=1)

        max_explanation_features = np.max(sparsity_scores)
        
        # Verify that no explanation uses more than 17 features (as stated in the paper)
        assert max_explanation_features <= 17, \
            f"Explanations should use at most 17 features, got {max_explanation_features}"