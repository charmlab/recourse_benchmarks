import numpy as np
import torch


def hval_xgb(xgb_clf, ystar, x):
    match_ = (ystar == xgb_clf.predict(x.detach().cpu())) * 1.0
    return match_.sum() / len(match_)


def hval_clf(ann_clf, ystar, x):
    match_ = (ystar == 1.0 * (ann_clf(x) > 0.5).detach().cpu().squeeze()) * 1.0
    return match_.sum().item() / len(match_)


def sval_sk(rf_clf, ystar, x):
    return rf_clf.predict_proba(x)[:, ystar].mean()


def sval_ann(ann_clf, ystar, x):
    if ystar == 0:
        return 1 - ann_clf(x).mean().item()
    else:
        return ann_clf(x).mean().item()


def logval_sk(rf_clf, ystar, x):
    return np.log(rf_clf.predict_proba(x)[:, ystar]).mean()


def cost_fn(x, xf, p=1):
    """
    (Bxd, Bxd)  - > (B,1)
    """
    return torch.norm(x - xf, dim=1, p=p, keepdim=True)


def get_neg_inst(ann_clf, X, ystar, sample=None):
    for param in ann_clf.parameters():
        DEVICE = param.device
        break
    # logging.info(f" Using device:{DEVICE}")
    xf = X.to(DEVICE)
    predf = ann_clf(xf)
    yf = ((predf > 0.5) * 1.0).squeeze().detach()
    xf_r = xf[[yf != ystar]]

    leng = len(xf_r)
    if sample is None or sample > leng:
        return xf_r
    else:
        return xf_r[np.random.choice(leng, sample, replace=False)]
        # need to implement subsampling, get indices from original X tensor to aid evaluations
