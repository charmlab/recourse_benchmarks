import pandas as pd
from utils import *
from care.distance_obj import distanceObj

def evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                            toolbox, objective_names, featureScaler, feature_names):

    # merging x with counterfactuals
    n_cf = cfs_ord.shape[0]
    x_ord = pd.DataFrame(data=x_ord.reshape(1, -1), columns=feature_names)
    x_cfs_ord = pd.concat([x_ord, cfs_ord])
    x_cfs_ord.reset_index(drop=True, inplace=True)

    # mapping data to required formats for evaluation and prediction
    x_cfs_theta = ord2theta(x_cfs_ord, featureScaler)
    x_cfs_ohe = ord2ohe(x_cfs_ord.to_numpy(), dataset)

    # evaluating counterfactuals
    evaluation = np.asarray([np.asarray(toolbox.evaluate(ind)) for ind in x_cfs_theta])

    # checking the validity counterfactulas (i.e., correct prediction)
    # validity of every individual counterfactual
    i_validity = (evaluation[:, 0] == 0.0).astype(int)
    # average validity of a counterfactual set
    s_validity = [np.mean(i_validity[1:]) for _ in range(n_cf + 1)]

    # measuring the diversity of of counterfacutuals
    if n_cf == 1:
        similar_features = 1.0
        similar_values = 1.0
        distance = 0.0
    else:
        # finding changed features in every counterfactual
        changed_feature = []
        for i in range(n_cf):
            idx_changed = np.where(x_cfs_ord.iloc[0] != x_cfs_ord.iloc[i + 1])[0]
            change_dict = {}
            for j in idx_changed:
                change_dict[dataset['feature_names'][j]] =  x_cfs_ord.iloc[i + 1, j]
            changed_feature.append(change_dict)

        # feature-based diversity: measuring Jaccard coefficient for changed features of every pair of counterfactuals
        similar_features = []
        for i in range(0, n_cf - 1):
            for j in range(i + 1, n_cf):
                if changed_feature[i] == {}:
                    pass
                else:
                    similarity = len(set(changed_feature[i]) & set(changed_feature[j])) / \
                                 len(set(changed_feature[i]) | set(changed_feature[j]))
                    similar_features.append(similarity)
        similar_features = np.mean(similar_features)

        # value-based diversity: measuring value similarity for changed features of every pair of counterfactuals
        similar_values = []
        for i in range(0, n_cf - 1):
            for j in range(i + 1, n_cf):
                if changed_feature[i] == {}:
                    pass
                else:
                    similarity = [1 if changed_feature[i][f] == changed_feature[j][f] else 0
                                 for f in set(changed_feature[i]) & set(changed_feature[j])]
                    if similarity == []:
                        pass
                    else:
                        similarity = np.mean(similarity)
                        similar_values.append(similarity)
        similar_values = np.mean(similar_values)

        # distance-based diversity: measuring distance between every pair of counterfactuals using Gower metric
        distance = []
        for i in range(0, n_cf-1):
            for j in range(i+1, n_cf):
                distance.append(distanceObj(cfs_ord.iloc[i].to_numpy(), cfs_ord.iloc[j].to_numpy(),
                                            dataset['feature_width'], dataset['continuous_indices'],
                                            dataset['discrete_indices']))
        distance = np.mean(distance)

    f_diversity = [(1.0 - similar_features) for _ in range(n_cf + 1)]
    v_diversity = [(1.0 - similar_values) for _ in range(n_cf + 1)]
    d_diversity = [distance for _ in range(n_cf + 1)]

    # creating an evaluation data frame
    if task == 'classification':
        label = predict_fn(x_cfs_ohe)
        probability = np.max(predict_proba_fn(x_cfs_ohe), axis=1)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation, i_validity, s_validity, f_diversity, v_diversity, d_diversity, label, probability],
                                  columns=objective_names+['i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity', 'Class', 'Probability'])
    else:
        response = predict_fn(x_cfs_ohe)
        x_cfs_eval = pd.DataFrame(data=np.c_[evaluation, i_validity, s_validity, f_diversity, v_diversity, d_diversity, response],
                                  columns=objective_names+['i-Validity', 's-Validity', 'f-Diversity', 'v-Diversity', 'd-Diversity', 'Response'])

    # creating data frames for counterfactuals
    cfs_ord_ = x_cfs_ord.iloc[1:,:].copy(deep=True)
    cfs_eval = x_cfs_eval.iloc[1:,:].copy(deep=True)

    # indexing counterfactual data frames
    index = pd.Series(['cf_'+str(i) for i in range(len(cfs_eval))])
    cfs_ord_ = cfs_ord_.set_index(index)
    cfs_eval = cfs_eval.set_index(index)

    # indexing x+counterfactual data frames
    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_ord) - 1)])
    x_cfs_ord = x_cfs_ord.set_index(index)
    x_cfs_eval = x_cfs_eval.set_index(index)

    return cfs_ord_, cfs_eval, x_cfs_ord, x_cfs_eval