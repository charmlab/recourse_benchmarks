import numpy as np

def ord2ohe(X_ord, dataset):
    continuous_availability = dataset['continuous_availability']
    discrete_availability = dataset['discrete_availability']
    ohe_feature_encoder = dataset['ohe_feature_encoder']
    len_continuous_ord = dataset['len_continuous_ord']
    len_discrete_ord = dataset['len_discrete_ord']

    if X_ord.shape.__len__() == 1:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete.reshape(1,-1)).ravel()
            X_ohe = np.r_[X_continuous, X_discrete]
            return X_ohe
        elif continuous_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_ohe = X_continuous.copy()
            return X_ohe
        elif discrete_availability:
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete.reshape(1, -1)).ravel()
            X_ohe = X_discrete.copy()
            return X_ohe
    else:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete)
            X_ohe = np.c_[X_continuous,X_discrete]
            return X_ohe
        elif continuous_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_ohe = X_continuous.copy()
            return X_ohe
        elif discrete_availability:
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ohe_feature_encoder.transform(X_discrete)
            X_ohe = X_discrete.copy()
            return X_ohe

def ohe2ord(X_ohe, dataset):
    continuous_availability = dataset['continuous_availability']
    discrete_availability = dataset['discrete_availability']
    ohe_feature_encoder = dataset['ohe_feature_encoder']
    len_continuous_ohe = dataset['len_continuous_ohe']
    len_discrete_ohe = dataset['len_discrete_ohe']

    if X_ohe.shape.__len__() == 1:
        if continuous_availability and discrete_availability:
            X_continuous = X_ohe[len_continuous_ohe[0]:len_continuous_ohe[1]]
            X_discrete = X_ohe[len_discrete_ohe[0]:len_discrete_ohe[1]]
            X_discrete = ohe_feature_encoder.inverse_transform(X_discrete.reshape(1,-1)).ravel()
            X_ord = np.r_[X_continuous, X_discrete]
            return X_ord
        elif continuous_availability:
            X_continuous = X_ohe[len_continuous_ohe[0]:len_continuous_ohe[1]]
            X_ord = X_continuous.copy()
            return X_ord
        elif discrete_availability:
            X_discrete = X_ohe[len_discrete_ohe[0]:len_discrete_ohe[1]]
            X_discrete = ohe_feature_encoder.inverse_transform(X_discrete.reshape(1,-1)).ravel()
            X_ord = X_discrete.copy()
            return X_ord
    else:
        if continuous_availability and discrete_availability:
            X_continuous = X_ohe[:,len_continuous_ohe[0]:len_continuous_ohe[1]]
            X_discrete = X_ohe[:,len_discrete_ohe[0]:len_discrete_ohe[1]]
            X_discrete = ohe_feature_encoder.inverse_transform(X_discrete)
            X_ord = np.c_[X_continuous,X_discrete]
            return X_ord
        elif continuous_availability:
            X_continuous = X_ohe[:,len_continuous_ohe[0]:len_continuous_ohe[1]]
            X_ord = X_continuous.copy()
            return X_ord
        elif discrete_availability:
            X_discrete = X_ohe[:,len_discrete_ohe[0]:len_discrete_ohe[1]]
            X_discrete = ohe_feature_encoder.inverse_transform(X_discrete)
            X_ord = X_discrete.copy()
            return X_ord

def org2ord(X_org, dataset):
    continuous_availability = dataset['continuous_availability']
    discrete_availability = dataset['discrete_availability']
    num_feature_scaler = dataset['num_feature_scaler']
    ord_feature_encoder = dataset['ord_feature_encoder']
    len_continuous_org = dataset['len_continuous_org']
    len_discrete_org = dataset['len_discrete_org']

    if X_org.shape.__len__() == 1:
        if continuous_availability and discrete_availability:
            X_continuous = X_org[len_continuous_org[0]:len_continuous_org[1]]
            X_continuous = num_feature_scaler.transform(X_continuous.reshape(1, -1)).ravel()
            X_discrete = X_org[len_discrete_org[0]:len_discrete_org[1]]
            X_discrete = ord_feature_encoder.transform(X_discrete.reshape(1,-1)).ravel()
            X_ord = np.r_[X_continuous, X_discrete]
            return X_ord
        elif continuous_availability:
            X_continuous = X_org[len_continuous_org[0]:len_continuous_org[1]]
            X_continuous = num_feature_scaler.transform(X_continuous.reshape(1, -1)).ravel()
            X_ord = X_continuous.copy()
            return X_ord
        elif discrete_availability:
            X_discrete = X_org[len_discrete_org[0]:len_discrete_org[1]]
            X_discrete = ord_feature_encoder.transform(X_discrete.reshape(1,-1)).ravel()
            X_ord = X_discrete.copy()
            return X_ord
    else:
        if continuous_availability and discrete_availability:
            X_continuous = X_org[:,len_continuous_org[0]:len_continuous_org[1]]
            X_continuous = num_feature_scaler.transform(X_continuous)
            X_discrete = X_org[:,len_discrete_org[0]:len_discrete_org[1]]
            X_discrete = ord_feature_encoder.transform(X_discrete)
            X_ord = np.c_[X_continuous,X_discrete]
            return X_ord
        elif continuous_availability:
            X_continuous = X_org[:,len_continuous_org[0]:len_continuous_org[1]]
            X_continuous = num_feature_scaler.transform(X_continuous)
            X_ord = X_continuous.copy()
            return X_ord
        elif discrete_availability:
            X_discrete = X_org[:,len_discrete_org[0]:len_discrete_org[1]]
            X_discrete = ord_feature_encoder.transform(X_discrete)
            X_ord = X_discrete.copy()
            return X_ord

def ord2org(X_ord, dataset):
    continuous_availability = dataset['continuous_availability']
    discrete_availability = dataset['discrete_availability']
    num_feature_scaler = dataset['num_feature_scaler']
    ord_feature_encoder = dataset['ord_feature_encoder']
    len_continuous_ord = dataset['len_continuous_ord']
    len_discrete_ord = dataset['len_discrete_ord']
    continuous_precision = dataset['continuous_precision']

    if X_ord.shape.__len__() == 1:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_continuous = num_feature_scaler.inverse_transform(X_continuous.reshape(1, -1)).ravel()
            for f, dec in enumerate(continuous_precision):
                X_continuous[f] = np.around(X_continuous[f], decimals=dec)
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ord_feature_encoder.inverse_transform(X_discrete.reshape(1,-1)).ravel()
            X_org = np.r_[X_continuous, X_discrete]
            return X_org
        elif continuous_availability:
            X_continuous = X_ord[len_continuous_ord[0]:len_continuous_ord[1]]
            X_continuous = num_feature_scaler.inverse_transform(X_continuous.reshape(1, -1)).ravel()
            for f, dec in enumerate(continuous_precision):
                X_continuous[f] = np.around(X_continuous[f], decimals=dec)
            X_org = X_continuous.copy()
            return X_org
        elif discrete_availability:
            X_discrete = X_ord[len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ord_feature_encoder.inverse_transform(X_discrete.reshape(1,-1)).ravel()
            X_org = X_discrete.copy()
            return X_org
    else:
        if continuous_availability and discrete_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_continuous = num_feature_scaler.inverse_transform(X_continuous)
            for f, dec in enumerate(continuous_precision):
                X_continuous[:,f] = np.around(X_continuous[:,f], decimals=dec)
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ord_feature_encoder.inverse_transform(X_discrete)
            X_org = np.c_[X_continuous,X_discrete]
            return X_org
        elif continuous_availability:
            X_continuous = X_ord[:,len_continuous_ord[0]:len_continuous_ord[1]]
            X_continuous = num_feature_scaler.inverse_transform(X_continuous)
            for f, dec in enumerate(continuous_precision):
                X_continuous[:,f] = np.around(X_continuous[:,f], decimals=dec)
            X_org = X_continuous.copy()
            return X_org
        elif discrete_availability:
            X_discrete = X_ord[:,len_discrete_ord[0]:len_discrete_ord[1]]
            X_discrete = ord_feature_encoder.inverse_transform(X_discrete)
            X_org = X_discrete.copy()
            return X_org

def ord2theta(X_ord, featureScaler):
    if X_ord.shape.__len__() == 1:
        X_theta = featureScaler.transform(X_ord.reshape(1,-1)).ravel()
        return X_theta
    else:
        X_theta = featureScaler.transform(X_ord)
        return X_theta

def theta2ord(X_theta, featureScaler, dataset):
    discrete_availability = dataset['discrete_availability']
    discrete_indices = dataset['discrete_indices']
    if X_theta.shape.__len__() == 1:
        X_ord = featureScaler.inverse_transform(X_theta.reshape(1,-1)).ravel()
        if discrete_availability:
            X_ord[discrete_indices] = np.rint(X_ord[discrete_indices])
        return X_ord
    else:
        X_ord = featureScaler.inverse_transform(X_theta)
        if discrete_availability:
            X_ord[:, discrete_indices] = np.rint(X_ord[:, discrete_indices])
        return X_ord

def theta2org(X_theta, featureScaler, dataset):
    X_ord = theta2ord(X_theta, featureScaler, dataset)
    X_org = ord2org(X_ord, dataset)
    return X_org