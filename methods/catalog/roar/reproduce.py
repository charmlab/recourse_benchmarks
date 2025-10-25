import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from data.catalog.online_catalog import DataCatalog
from methods.catalog.roar.model import Roar
from models.catalog.catalog import ModelCatalog

RANDOM_SEED = 54321

# Get the absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).parent

# Construct the path to your data directory, relative to the script location
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data" / "catalog" / "_data_main" / "raw_data"


#Find indices where recourse is needed
def recourse_needed(predict_fn, X, target=1):
	return np.where(predict_fn(X) == 1-target)[0]

#Recourse validity
def recourse_validity(predict_fn, rs, target=0.5): # think it needs to 0.5 cause predict gives prob
	return sum(predict_fn(rs)>target)/len(rs)

#Functions to compute the cost of recourses
def l1_cost(xs, rs):
    # ensure numpy arrays
    xs_arr = xs.to_numpy()
    rs_arr = rs.to_numpy()

    print(xs_arr)
    print("-----------------------------------")
    print(rs_arr)
    
    # compute row-wise L1 norms
    cost = np.linalg.norm(rs_arr - xs_arr, ord=1, axis=1)
    
    return np.mean(cost)

# because running the experiment just the same as the paper would require breaking tweaks to the implemented ROAR
# the reproduce will be a partial one, showing us the validation results just on the first models (M1)s of the paper


#TODO make sure to update mlmodel_catalog.yaml
#TODO make sure to update reproduce to match above statement, get permission for this as well.
#TODO add the SBA and Student datasets and make them accessible

@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("sba", "linear", "pytorch"),
        ("sba", "mlp", "pytorch"),
    ],
)
def test_roar(dataset_name, model_type, backend):
    
    # results = {}

    args = {
        'cost': 'l1',
        'lamb': 0.1,
    }
    
    # for i in range(args['n_trials']): Wont do trials, just one run
    # print("Trail %d" % i)
    
    # results_i = {}
    # fold = i
    
    # print("loading %s dataset" % args['data']) # TODO update this to use the existing pipeline, not the code above
    # if args["data"] == 'correction':
    #     data_name = 'german'
    #     #data1, data2 = data.get_data()
    # elif args["data"] == "temporal":
    #     data_name = 'sba'
    #     #data1, data2 = data.get_data()
    # elif args["data"] == "geospatial":
    #     data_name = 'student'
    #     #data1, data2 = data.get_data()
    
    # X1_train, y1_train, X1_test, y1_test = data1 # unsure if we acually recive data like this, may need changing
    # X2_train, y2_train, X2_test, y2_test = data2
    
    print("Training %s models" % model_type) #TODO as compromise, I will only run on m1(s)
    if model_type == "linear":
        data = DataCatalog(dataset_name, model_type='linear', train_split=0.8)
        m1 = ModelCatalog(data, model_type="linear", backend=backend) # m1 = LR()
        data2 = DataCatalog(dataset_name+"_modified", model_type='linear', train_split=0.8)
        m2 = ModelCatalog(data2, model_type="linear", backend=backend)
    if model_type == "mlp":
        data = DataCatalog(dataset_name, model_type='mlp', train_split=0.8)
        m1 = ModelCatalog(data, model_type="mlp", backend=backend)# m1 = NN(X1_train.shape[1])
        data2 = DataCatalog(dataset_name+"_modified", model_type='mlp', train_split=0.8)
        m2 = ModelCatalog(data2, model_type="mlp", backend=backend)
        # m2 = NN(X1_train.shape[1])

    # print(data._df_train)
    m1._test_accuracy() #TODO best practice to remove, as they are private
    m2._test_accuracy()

    print("Using %s cost" % args['cost'])
    if args['cost'] == "l1":
        feature_costs = None
    
    coefficients=intercept=None

    # if args['base_model'] != "nn":
    #     coefficients=m1.sklearn_model.coef_[0]
    #     intercept = m1.sklearn_model.intercept_
    
    roar = Roar(mlmodel=m1, hyperparams={}, coeffs=coefficients, intercepts=intercept)          

    lamb = args['lamb']

    recourses=[]
    deltas=[]

    factuals = (data._df_test).sample(n=10, random_state=RANDOM_SEED)

    factuals = factuals.drop('y', axis=1)

    r = roar.get_counterfactuals(factuals=factuals)
    recourses.append(r)
    
    # results_i["recourses"] = recourses

    recourses = np.array(recourses)
    print(factuals.columns)
    print(r.columns)

    # For model 1, ran on og dataset
    m1_validity = recourse_validity(m1.predict, r)
    # results_i["m1_validity"] = m1_validity
    print("M1 validity: %f" % m1_validity)

    # For model 2, ran on modified dataset
    m2_validity = recourse_validity(m2.predict, r)
    # results_i["m2_validity"] = m2_validity
    print("M2 validity: %f" % m2_validity)

    if args['cost'] == "l1":
        cost = l1_cost(factuals, r)
        
    # results_i["cost"] = cost
    print("%s cost: %f" % (args['cost'], cost))

    assert m1_validity >= 0.9
    assert m2_validity >= 0.9

    # results[i] = results_i

    # results_i["recourses"] = recourses

    # -------------- end of for loop ----------------------------

    # agg_m1_validity = []
    # agg_m2_validity = []
    # agg_cost = []
    # for i in range(args['n_trials']):
    #     agg_m1_validity.append(results[i]["m1_validity"])
    #     agg_m2_validity.append(results[i]["m2_validity"])
    #     agg_cost.append(results[i]["cost"])

    # print("Average M1 validity: %f +- %f" % (np.mean(agg_m1_validity), np.std(agg_m1_validity)))
    # print("Average M2 validity: %f +- %f" % (np.mean(agg_m2_validity), np.std(agg_m2_validity)))
    # print("Average cost: %f +- %f" % (np.mean(agg_cost), np.std(agg_cost)))

if __name__ == '__main__':
    test_roar("sba", "linear", "pytorch")