import pandas as pd
from utils import *

def recoverOriginals(x_ord, cfs_ord, dataset, feature_names):

    # mapping data to original format
    x_org = ord2org(x_ord, dataset)
    x_org = pd.DataFrame(data=x_org.reshape(1,-1), columns=feature_names)
    cfs_org = ord2org(cfs_ord.to_numpy(), dataset)
    cfs_org = pd.DataFrame(data=cfs_org, columns=feature_names)

    # merging x with counterfactuals
    x_cfs_org= pd.concat([x_org, cfs_org])

    # indexing x+counterfactual data frames
    index = pd.Series(['x'] + ['cf_' + str(i) for i in range(len(x_cfs_org) - 1)])
    x_cfs_org = x_cfs_org.set_index(index)

    # highlighting changed features in counterfactuals
    x_cfs_highlight = x_cfs_org.copy(deep=True)
    for f in range(x_cfs_org.shape[1]):
        ind = np.where(x_cfs_org.iloc[:,f] == x_cfs_org.iloc[0,f])[0]
        x_cfs_highlight.iloc[ind[1:],f] = '_'

    return x_org, cfs_org, x_cfs_org, x_cfs_highlight