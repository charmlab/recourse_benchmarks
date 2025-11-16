# in order to accurately reproduce the work in the paper
# i will run the experiments similar to them, but I 
# will not make use of the carla model method directly, but rather run 
# a modified version of the run_method funtion found in larr,py

import pytest


@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("german", "mlp", "pytorch"),
    ],
)
def run_experiment(dataset_name, model_type, backend):
    ...