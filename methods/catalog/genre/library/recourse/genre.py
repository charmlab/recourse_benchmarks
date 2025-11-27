import torch


class GenRe:
    def __init__(self, pair_model, temp, sigma, best_k, ann_clf, ystar, cat_mask):
        self.pair_model = pair_model.eval()
        self.temp = temp
        self.sigma = sigma
        # assert 0 <sigma < 1/pair_model.n_bins
        self.best_k = best_k
        self.model = ann_clf.eval()
        self.ystar = ystar
        self.cat_mask = cat_mask

    @torch.no_grad()
    def __call__(self, xf_r):
        DEVICE = xf_r.device
        yf_r = torch.ones(xf_r.shape[0]).to(DEVICE) * self.ystar
        self.pair_model = self.pair_model.to(DEVICE)
        self.model = self.model.to(DEVICE)

        sampled_list = []
        for i in range(self.best_k):
            sample_xcf = self.pair_model._sample(
                xf_r, yf_r, y=yf_r * 0 + self.ystar, temp=self.temp, sigma=self.sigma
            )
            # project categorical variables
            sample_xcf[:, self.cat_mask] = torch.round(sample_xcf[:, self.cat_mask])
            if self.best_k == 1:
                return sample_xcf
            sampled_list.append(sample_xcf.detach().unsqueeze(1))

        sample_concat = torch.cat(sampled_list, dim=1)
        sample_predictions = self.model(sample_concat).squeeze()
        best_sample_idx = sample_predictions.argmax(dim=1)

        # print(f"[DEBUG GENRE] xf_r.shape: {xf_r.shape}")
        # print(f"[DEBUG GENRE] sample_concat.shape: {sample_concat.shape}")
        # print(f"[DEBUG GENRE] best_sample_idx.shape: {best_sample_idx.shape}")
        # print(f"[DEBUG GENRE] best_sample_idx: {best_sample_idx}")
        # print(f"[DEBUG GENRE] torch.arange(xf_r.shape[0]).shape: {torch.arange(xf_r.shape[0]).shape}")

        return sample_concat[torch.arange(xf_r.shape[0]), best_sample_idx, :]
