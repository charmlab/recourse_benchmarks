from data.catalog import DataCatalog
from evaluation import Benchmark
import evaluation.catalog as evaluation_catalog
from models.catalog import ModelCatalog
from random import seed
from recourse_methods import GrowingSpheres

RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again

# load a catalog dataset
data_name = "adult"
dataset = DataCatalog(data_name, "mlp", 0.8)

# load artificial neural network from catalog
model = ModelCatalog(dataset, "mlp", "tensorflow")

# get factuals from the data to generate counterfactual examples
factuals = (dataset._df_train).sample(n=10, random_state=RANDOM_SEED)

# load a recourse model and pass black box model
gs = GrowingSpheres(model)

# generate counterfactual examples
counterfactuals = gs.get_counterfactuals(factuals)

# Generate Benchmark for recourse method, model and data
benchmark = Benchmark(model, gs, factuals)
evaluation_measures = [
      evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
      evaluation_catalog.Distance(benchmark.mlmodel),
      evaluation_catalog.SuccessRate(),
      evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
      evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
      evaluation_catalog.AvgTime({"time": benchmark.timer}),
]
df_benchmark = benchmark.run_benchmark(evaluation_measures)
print(df_benchmark)