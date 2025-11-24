def proximityObj(cf_ohe, proximity_model):
    status = proximity_model.predict(cf_ohe.reshape(1, -1))[0]
    fitness = max(0, status)
    return fitness
