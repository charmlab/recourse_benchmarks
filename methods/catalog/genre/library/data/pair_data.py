import numpy as np
import torch
from torch.utils.data import Dataset


class PairData(Dataset):
    def __init__(self, X, y, ystar, k=2, dist_p=2):
        self.X = X
        self.y = y
        self.ystar = ystar

        self.pair_idxs = None
        self.k = k
        self.dist_p = dist_p
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.X, self.X, p=self.dist_p)
        # TODO: do we need to avoid self-pairing?
        D = D + torch.eye(D.shape[0]) * 1e10  # avoid self-pairing

        # TODO: should only output good labels?
        good_idxs = torch.where(self.y == self.ystar)[0]
        self.pair_idxs = torch.topk(D[:, good_idxs], k=self.k, largest=False)[1]
        self.pair_idxs = good_idxs[self.pair_idxs]
        pass

    def __len__(self):
        return len(self.X) * self.k

    def __getitem__(self, idx):
        # pair_id = self.pair_idxs[idx][torch.randperm(len(self.pair_idxs[idx]))[0]]
        idx, pair_id = idx // self.k, self.pair_idxs[idx // self.k][idx % self.k]
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "pair_x": self.X[pair_id],
            "pair_y": self.y[pair_id],
        }


class PairDatav2(Dataset):
    def __init__(self, X, y, ystar, k=2, dist_p=2):
        self.X = X
        self.y = y
        self.ystar = ystar
        self.goodX = X[y == ystar]
        self.goody = y[y == ystar]
        self.pair_idxs = None
        self.k = k
        self.dist_p = dist_p
        self.create_pairs()

    def create_pairs(self):
        # allows for self pairing, it's ok
        D = torch.cdist(self.goodX, self.X, p=self.dist_p)
        self.pair_idxs = torch.topk(D, k=self.k, largest=False)[1]

    def __len__(self):
        return len(self.goodX) * self.k

    def __getitem__(self, idx):
        # pair_id = self.pair_idxs[idx][torch.randperm(len(self.pair_idxs[idx]))[0]]
        idx, pair_idx = idx // self.k, self.pair_idxs[idx // self.k][idx % self.k]
        return {
            "x": self.X[pair_idx],
            "y": self.y[pair_idx],
            "pair_x": self.goodX[idx],
            "pair_y": self.goody[idx],
        }


class WeightedPairData(Dataset):
    def __init__(self, X, y, ystar, lambda_=0.9, k=2, p=2):
        src_labels = [1 - ystar]
        tgt_labels = [ystar]

        if len(src_labels) == 2:
            self.src_X = X
            self.src_y = y
        else:
            self.src_X = X[y == src_labels[0]]
            self.src_y = y[y == src_labels[0]]

        if len(tgt_labels) == 2:
            self.tgt_X = X
            self.tgt_y = y
        else:
            self.tgt_X = X[y == tgt_labels[0]]
            self.tgt_y = y[y == tgt_labels[0]]

        self.ystar = ystar

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)
        topk_out = torch.topk(D, k=self.k, largest=False)
        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X) * self.k

    def __getitem__(self, idx):

        idx, pair_idx, wght_idx = (
            idx // self.k,
            self.pair_idxs[idx // self.k][idx % self.k],
            idx % self.k,
        )

        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_idx],
            "pair_y": self.tgt_y[pair_idx],
            "weight": self.pair_prob[idx][wght_idx],
        }


# TODO: Deprecate StochasticPairData
class StochasticPairData(Dataset):
    def __init__(self, X, y, ystar, lambda_=0.9, k=2, p=2):
        src_labels = [1 - ystar]
        tgt_labels = [ystar]

        if len(src_labels) == 2:
            self.src_X = X
            self.src_y = y
        else:
            self.src_X = X[y == src_labels[0]]
            self.src_y = y[y == src_labels[0]]

        if len(tgt_labels) == 2:
            self.tgt_X = X
            self.tgt_y = y
        else:
            self.tgt_X = X[y == tgt_labels[0]]
            self.tgt_y = y[y == tgt_labels[0]]

        self.ystar = ystar

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }


class StochasticPairs(Dataset):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, lambda_=0.9, k=2, p=2):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }


class StochasticPairsNegSamp(Dataset):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, num_neg, lambda_, k, p):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.num_neg = num_neg
        self.tgt_len = len(self.tgt_X)
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        neg_pair_id = list(
            np.random.choice(self.tgt_len, self.num_neg, replace=False)
        )  # multiple, for each example we sample num_neg negative pairs
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
            "neg_pair_x": self.tgt_X[neg_pair_id],
        }


class StochasticPairsImmut(Dataset):
    def __init__(
        self, src_X, tgt_X, src_y, tgt_y, immutable_mask, lambda_=0.9, k=2, p=2
    ):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        if torch.all(torch.tensor(immutable_mask)):
            self.immutable_mask = None
        else:
            self.immutable_mask = immutable_mask
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        expD = torch.exp(-self.lambda_ * D)

        if self.immutable_mask is not None:
            immut_dist = 1 - 1.0 * (
                torch.cdist(
                    self.src_X[:, self.immutable_mask],
                    self.tgt_X[:, self.immutable_mask],
                    p=self.p,
                )
                > 0
            )
            expD = immut_dist * expD

        if self.k > D.shape[1]:
            topk_out = torch.topk(expD, k=D.shape[1], largest=True)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(expD, k=self.k, largest=True)

        pair_dist_exp = topk_out[0]

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }
