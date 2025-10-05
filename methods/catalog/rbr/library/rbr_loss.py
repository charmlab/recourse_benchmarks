# methods/catalog/rbr/library.py
import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from sklearn.utils import check_random_state


# ---------- low-level helpers & projections ----------

@torch.no_grad()
def l2_projection(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Euclidean projection onto an L2-ball for last axis.
    x: shape (..., d)
    radius: scalar
    """
    norm = torch.linalg.norm(x, ord=2, axis=-1)
    # avoid divide by zero
    denom = torch.maximum(norm, torch.tensor(radius, device=x.device))
    scale = (radius / denom).unsqueeze(-1)
    return scale * x

# in the original code but never used
# def reconstruct_encoding_constraints(x: torch.Tensor, cat_pos: Optional[Sequence[int]]):
#     """
#     Round/clamp categorical encodings (binary) at positions in cat_pos.
#     """
#     if cat_pos is None:
#         return x
#     x_enc = x.clone()
#     for pos in cat_pos:
#         x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
#     return x_enc


# ---------- likelihood modules ----------

class OptimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim: torch.Tensor, epsilon_op: torch.Tensor, sigma: torch.Tensor, device: torch.device):
        super().__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone()
        v = torch.maximum(v, torch.tensor(0.0, device=self.device))
        result = l2_projection(v, float(self.epsilon_op))
        return result.to(self.device)

    def _forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim
        L = torch.log(d) + (c - v[..., 0]) ** 2 / (2 * d ** 2) + (p - 1) * torch.log(self.sigma)
        return L

    def forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = torch.log(d) + (c - v[..., 0]) ** 2 / (2 * d ** 2) + (p - 1) * torch.log(self.sigma)

        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d ** 2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d ** 3

        return L, v_grad

    def optimize(self, x: torch.Tensor, x_feas: torch.Tensor, max_iter: int = int(1e3), verbose: bool = False):
        v = torch.zeros([x_feas.shape[0], 2], device=self.device)
        lr = 1.0 / math.sqrt(max_iter)

        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(v, x.expand([x_feas.shape[0], -1]), x_feas)
            v = self.projection(v - lr * grad)

            loss_sum = F.sum().item()
            loss_diff = min_loss - loss_sum
            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
            if verbose and (t % 200 == 0):
                print(f"[Optimistic] iter {t} loss {loss_sum:.6f}")
        return v


class PessimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim: torch.Tensor, epsilon_pe: torch.Tensor, sigma: torch.Tensor, device: torch.device):
        super().__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clone()
        u = torch.maximum(u, torch.tensor(0.0, device=self.device))
        result = l2_projection(u, float(self.epsilon_pe) / math.sqrt(float(self.x_dim)))
        return result.to(self.device)

    def _forward(self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p)
        inside = (zeta + self.epsilon_pe ** 2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (p - 1)
        f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))

        L = -torch.log(d) - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d ** 2) - (p - 1) * torch.log(f + self.sigma)
        return L

    def forward(self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p)
        inside = (zeta + self.epsilon_pe ** 2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (p - 1)
        f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))

        L = -torch.log(d) - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d ** 2) - (p - 1) * torch.log(f + self.sigma)

        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d ** 2 - (p * u[..., 0]) / (f * (f + self.sigma))
        u_grad[..., 1] = -1 / d + (c + sqrt_p * u[..., 0]) ** 2 / d ** 3 + u[..., 1] / (f * (f + self.sigma))

        return L, u_grad

    def optimize(self, x: torch.Tensor, x_feas: torch.Tensor, max_iter: int = int(1e3), verbose: bool = False):
        u = torch.zeros([x_feas.shape[0], 2], device=self.device)
        lr = 1.0 / math.sqrt(max_iter)

        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(u, x.expand([x_feas.shape[0], -1]), x_feas)
            u = self.projection(u - lr * grad)

            loss_sum = F.sum().item()
            loss_diff = min_loss - loss_sum
            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
            if verbose and (t % 200 == 0):
                print(f"[Pessimistic] iter {t} loss {loss_sum:.6f}")
        return u


# ---------- RBRLoss wrapper ----------

class RBRLoss(torch.nn.Module):
    def __init__(
        self,
        X_feas: torch.Tensor,
        X_feas_pos: torch.Tensor,
        X_feas_neg: torch.Tensor,
        epsilon_op: float,
        epsilon_pe: float,
        sigma: float,
        device: torch.device,
        verbose: bool = False,
    ):
        super(RBRLoss, self).__init__()
        self.device = device
        self.verbose = verbose

        self.X_feas = X_feas.to(self.device)
        self.X_feas_pos = X_feas_pos.to(self.device)
        self.X_feas_neg = X_feas_neg.to(self.device)

        self.epsilon_op = torch.tensor(epsilon_op, device=self.device)
        self.epsilon_pe = torch.tensor(epsilon_pe, device=self.device)
        self.sigma = torch.tensor(sigma, device=self.device)
        self.x_dim = torch.tensor(X_feas.shape[-1], device=self.device)

        self.op_likelihood = OptimisticLikelihood(self.x_dim, self.epsilon_op, self.sigma, self.device)
        self.pe_likelihood = PessimisticLikelihood(self.x_dim, self.epsilon_pe, self.sigma, self.device)

    def forward(self, x: torch.Tensor, verbose: bool = False):
        if verbose or self.verbose:
            print(f"N_neg: {self.X_feas_neg.shape}, N_pos: {self.X_feas_pos.shape}")

        # pessimistic part
        if self.X_feas_pos.shape[0] > 0:
            u = self.pe_likelihood.optimize(x.detach().clone().expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos, verbose=self.verbose)
            F_pe = self.pe_likelihood._forward(u, x.expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos)
            denom = torch.logsumexp(F_pe, -1)
        else:
            denom = torch.tensor(0.0, device=self.device)

        # optimistic part
        if self.X_feas_neg.shape[0] > 0:
            v = self.op_likelihood.optimize(x.detach().clone().expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg, verbose=self.verbose)
            F_op = self.op_likelihood._forward(v, x.expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg)
            numer = torch.logsumexp(-F_op, -1)
        else:
            numer = torch.tensor(0.0, device=self.device)

        result = numer - denom
        return result, denom, numer


# ---------- high-level RBR generator (callable used by CARLA wrapper) ----------

def robust_bayesian_recourse(
    raw_model: Any,
    x0: np.ndarray,
    cat_features_indices: Optional[Sequence[int]] = None,
    train_data: Optional[np.ndarray] = None,
    num_samples: int = 50,
    perturb_radius: float = 0.1,
    delta_plus: float = 0.0,
    sigma: float = 1.0,
    epsilon_op: float = 0.1,
    epsilon_pe: float = 0.1,
    max_iter: int = 500,
    device: str = "cpu",
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    High-level function that matches the CARLA library-call pattern.
    Parameters largely mirror the original code you provided.
    - raw_model: object with .predict(np.ndarray) -> labels/probs
    - x0: 1D numpy array (a single factual)
    - cat_features_indices: indices of encoded categorical features to clamp/round
    - train_data: numpy array (N, d) required (used to find boundary and feasible set)
    Returns counterfactual as numpy array same shape as x0.
    """
    # helper to call raw_model.predict consistently
    def predict_fn_np(arr: np.ndarray) -> np.ndarray:
        # raw_model might accept (n,d) and return probs or labels
        preds = raw_model.predict(arr)
        preds = np.asarray(preds)
        # convert to single-label 0/1 if probabilities provided
        if preds.ndim == 2 and preds.shape[1] > 1:
            return preds.argmax(axis=1)
        if preds.dtype.kind in ("f",):
            return (preds >= 0.5).astype(int)
        return preds.astype(int)
    
     # find boundary point between x0 and nearest opposite-label train point
    def dist(a: torch.Tensor, b: torch.Tensor):
        return torch.linalg.norm(a - b, ord=1, axis=-1)
    
    # feasible set sampled around x_b
    def uniform_ball(x: torch.Tensor, r: float, n: int, rng_state):
        rng_local = check_random_state(rng_state)
        d = x.shape[0]
        V = rng_local.randn(n, d)
        V = V / np.linalg.norm(V, axis=1).reshape(-1, 1)
        V = V * (rng_local.random(n) ** (1.0 / d)).reshape(-1, 1)
        V = V * r + x.cpu().numpy()
        return torch.from_numpy(V).float().to(dev)

    def simplex_projection(x, delta):
        """
        Euclidean projection on a positive simplex
        """
        (p,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        u, _ = torch.sort(x, descending=True)
        cssv = torch.cumsum(u, 0)
        rho = torch.nonzero(u * torch.arange(1, p + 1).to(device) > (cssv - delta))[-1, 0]
        theta = (cssv[rho] - delta) / (rho + 1.0)
        w = torch.clip(x - theta, min=0)
        return w

    def projection(x, delta):
        """
        Euclidean projection on an L1-ball
        """
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x

        proj = simplex_projection(x_abs, delta=delta)
        proj *= torch.sign(x)

        return proj

    # device selection
    if "cuda" in device and torch.cuda.is_available():
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    rng = check_random_state(random_state)

    if train_data is None:
        raise ValueError("train_data must be provided to robust_bayesian_recourse")

    # ------- Implementation of fit_instance() ------------------
    x0_t = torch.from_numpy(x0.copy()).float().to(dev)

    train_t = torch.tensor(train_data).float().to(dev)

    # training label vector
    train_label = torch.tensor(predict_fn_np(train_data)).to(dev)

    # -------- Implementation of find_x_boundary() ---------------
    # find nearest opposite label examples and search along line for boundary
    dists = dist(train_t, x0_t)
    order = torch.argsort(dists)
    x_label = predict_fn_np(x0.reshape(1, -1))[0]
    candidates = train_t[order[train_label[order] == (1 - x_label)]]
    best_x_b = None
    best_dist = torch.tensor(float("inf"), device=dev)
    for x_c in candidates:
        lambdas = torch.linspace(0, 1, 100, device=dev)
        for lam in lambdas:
            x_b = (1 - lam) * x0_t + lam * x_c
            label = predict_fn_np(x_b.unsqueeze(0).cpu().numpy())[0]
            if label == 1 - x_label:
                curdist = dist(x0_t, x_b)
                if curdist < best_dist:
                    best_x_b = x_b.detach().clone()
                    best_dist = curdist.detach().clone()
                break
    # ------------------ end of find_x_boundary() --------------------

    # if best_x_b is None:
    #     # fallback: nearest opposite neighbor directly
    #     opp_idx = (train_label == (1 - x_label)).nonzero(as_tuple=False)
    #     if opp_idx.shape[0] == 0:
    #         # can't find opposite label in train set -> return original
    #         return x0.copy()
    #     first_idx = opp_idx[0, 0].item()
    #     best_x_b = train_t[first_idx].detach().clone()
    #     best_dist = dist(x0_t, best_x_b)

    delta = best_dist + delta_plus

    X_feas = uniform_ball(best_x_b, perturb_radius, num_samples, rng).float().to(dev)

    # apply categorical clamping if requested
    # if cat_features_indices:
    #     for i in range(X_feas.shape[0]):
    #         X_feas[i] = reconstruct_encoding_constraints(X_feas[i], cat_features_indices)

    y_feas = torch.tensor(predict_fn_np(X_feas.cpu().numpy())).to(dev)

    if (y_feas == 1).any():
        X_feas_pos = X_feas[y_feas == 1].reshape([int((y_feas == 1).sum().item()), -1])
    else:
        X_feas_pos = torch.empty((0, X_feas.shape[1]), device=dev)

    if (y_feas == 0).any():
        X_feas_neg = X_feas[y_feas == 0].reshape([int((y_feas == 0).sum().item()), -1])
    else:
        X_feas_neg = torch.empty((0, X_feas.shape[1]), device=dev)

    # build loss wrapper
    loss_fn = RBRLoss(X_feas, X_feas_pos, X_feas_neg, epsilon_op, epsilon_pe, sigma, device=dev, verbose=verbose)

    # ---------------- Start of optimize() ----------------
    # optimization loop - same basic behaviour as original code.
    x_t = best_x_b.detach().clone()
    x_t.requires_grad_(True)

    min_loss = float("inf")
    num_stable_iter = 0
    max_stable_iter = 10
    step = 1.0 / math.sqrt(1e3)

    for t in range(max_iter):
        if x_t.grad is not None:
            x_t.grad.data.zero_()

        F, denom, numer = loss_fn(x_t)
        F_sum = F.sum()
        F_sum.backward()
        # if we left L1-ball, break
        if torch.ge(torch.linalg.norm((x_t.detach() - x0_t), ord=1), float(delta)):
            break

        with torch.no_grad():
            x_new = x_t - step * x_t.grad
            # x_new = (lambda a, b: (a if torch.abs(a).sum() <= b else (l2_projection(a.unsqueeze(0), b).squeeze())))(
            #     x_new - x0_t, float(delta)
            # )

            x_new = projection(x_new - x0_t, float(delta)) + x0_t

        # enforce categorical encodings rounding/clamping
        # if cat_features_indices:
        #     x_new = reconstruct_encoding_constraints(x_new, cat_features_indices)

        for i, e in enumerate(x_new.data):
            x_t.data[i] = e

        loss_sum = F_sum.item()
        loss_diff = min_loss - loss_sum

        if loss_diff <= 1e-10:
            num_stable_iter += 1
            if num_stable_iter >= max_stable_iter:
                break
        else:
            num_stable_iter = 0
        min_loss = min(min_loss, loss_sum)

    cf = x_t.detach().cpu().numpy().squeeze()

    # ----------------------------- end of optimize() -----------------------

    # final clamping for feature valid ranges [0,1] if raw_model expects that (user may want different behaviour)
    # NOTE: the CARLA wrapper can do final "check_counterfactuals" conversions; here we return raw vector
    return cf
