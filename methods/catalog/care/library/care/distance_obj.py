import numpy as np

def distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices):
    distance = []
    if continuous_indices is not None:
        for j in continuous_indices:
            distance.append((1/feature_width[j]) * abs(x_ord[j]-cf_ord[j]))
    if discrete_indices is not None:
        for j in discrete_indices:
            distance.append(int(x_ord[j] != cf_ord[j]))
    return np.mean(distance)