import numpy as np

def coherencyObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices, correlationModel):
    distance = []
    cf_ord_ = cf_ord.copy()
    delta = np.nonzero(x_ord-cf_ord)[0]
    for m in correlationModel:
        feature = m['feature']
        model = m['model']
        inputs = m['inputs']
        score = m['score']
        if feature in delta:
            cf_ord_[feature] = model.predict(cf_ord[inputs].reshape(1, -1))
            if feature in discrete_indices:
                distance.append(score * int(cf_ord[feature] != cf_ord_[feature]))
            elif feature in continuous_indices:
                distance.append(score * (1 / feature_width[feature]) * abs(cf_ord[feature] - cf_ord_[feature]))
    cost = 0 if distance == [] else np.sum(distance)
    return cost

