import sys
sys.path.append('../')

import torch

import utils

class NNR:
    def __init__(self,X:torch.Tensor,y:torch.Tensor, batch_size:int, p:int=1):
        self.X = X.clone()
        self.y = y.clone()
        self.batch_size = batch_size
        self.p = p# norm of cost
        # returns nearest neighbours for now but can potentially return k-nn with k as an expected argument

    @torch.no_grad()
    def __call__(self,xf_r,ystar,DEVICE):
        xf_r = xf_r.clone()
        train_positive_examples = self.X[self.y == ystar].to(DEVICE)
        nnr_dataset = torch.utils.data.TensorDataset(xf_r)
        nnr_loader = torch.utils.data.DataLoader(nnr_dataset, batch_size=self.batch_size, shuffle=False)
        xcf_nnr = []
        for batch in nnr_loader:
            bxf_r = batch[0].to(DEVICE)
            # TODO: only calculates p=1 for cost, extend to general cost norm
            dist_matrix_nnr = utils.get_pair_dist(bxf_r,train_positive_examples,p=self.p)
            bxcf_r = train_positive_examples[torch.argmin(dist_matrix_nnr, dim=1)]
            xcf_nnr.append(bxcf_r)
        xcf_nnr = torch.cat(xcf_nnr, dim=0)
        return xcf_nnr.detach().cpu()