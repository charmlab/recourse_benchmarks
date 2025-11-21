import numpy as np

def outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn, probability_thresh, cf_class, cf_range):
    if task == 'classification':
        cf_probability = predict_proba_fn(cf_ohe.reshape(1, -1))[0, cf_class]
        cost = np.max([0, probability_thresh - cf_probability])
        return cost
    else:
        cf_response = predict_fn(cf_ohe.reshape(1, -1))
        if np.logical_and(cf_response >= cf_range[0], cf_response <= cf_range[1]):
            cost = 0
        else:
            cost = min(abs(cf_response - cf_range))
        return cost