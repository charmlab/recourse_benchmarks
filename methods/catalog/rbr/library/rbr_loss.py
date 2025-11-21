# methods/catalog/rbr/library.py
import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
from sklearn.utils import check_random_state

"""
This code is largely ported over from the original authors codebase. 
Light restructuring and modifications have been made in order to make it compatible with CARLAs structure.

Original code can be found at: https://github.com/VinAIResearch/robust-bayesian-recourse
"""

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
    denom = torch.max(norm, torch.tensor(radius, device=x.device))
    scale = (radius / denom).unsqueeze(1)
    return scale * x

# In the original code but never seemed to be used
def reconstruct_encoding_constraints(x: torch.Tensor, cat_pos: Optional[Sequence[int]]):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc



# ---------- likelihood modules ----------


class OptimisticLikelihood(torch.nn.Module):
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_op: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone()
        v = torch.max(v, torch.tensor(0, device=self.device))
        result = l2_projection(v, float(self.epsilon_op))
        return result.to(self.device)

    def _forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim
        L = (
            torch.log(d)
            + (c - v[..., 0]) ** 2 / (2 * d**2)
            + (p - 1) * torch.log(self.sigma)
        )
        return L

    def forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = (
            torch.log(d)
            + (c - v[..., 0]) ** 2 / (2 * d**2)
            + (p - 1) * torch.log(self.sigma)
        )

        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d**2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d**3

        return L, v_grad

    def optimize(
        self,
        x: torch.Tensor,
        x_feas: torch.Tensor,
        max_iter: int = int(1e3),
        verbose: bool = False,
    ):
        v = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(v, x, x_feas)
            v = self.projection(v - lr * grad)

            loss_sum = F.sum().data.item()
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
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_pe: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clone()
        u = torch.max(u, torch.tensor(0, device=self.device))
        result = l2_projection(u, float(self.epsilon_pe) / math.sqrt(float(self.x_dim)))
        return result.to(self.device)

    def _forward(
        self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6
    ):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        # p = p.float()
        sqrt_p = torch.sqrt(p.float())

        inside = (zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (
            p - 1
        )
        # f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))
        f = torch.sqrt(inside)

        L = (
            -torch.log(d)
            - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2)
            - (p - 1) * torch.log(f + self.sigma)
        )
        return L

    def forward(
        self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6
    ):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim

        # p = p.float() # issue with support with int tensors when taking sqrt?

        sqrt_p = torch.sqrt(p.float())
        inside = (zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (
            p - 1
        )
        # f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))
        f = torch.sqrt(inside)

        L = (
            -torch.log(d)
            - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2)
            - (p - 1) * torch.log(f + self.sigma)
        )

        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d**2 - (
            p * u[..., 0]
        ) / (f * (f + self.sigma))
        u_grad[..., 1] = (
            -1 / d
            + (c + sqrt_p * u[..., 0]) ** 2 / d**3
            + u[..., 1] / (f * (f + self.sigma))
        )

        return L, u_grad

    def optimize(
        self,
        x: torch.Tensor,
        x_feas: torch.Tensor,
        max_iter: int = int(1e3),
        verbose: bool = False,
    ):
        u = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1.0 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(u, x, x_feas)
            u = self.projection(u - lr * grad)

            loss_sum = F.sum().data.item()
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

        # print("This is epsilon op: ", self.epsilon_op)
        # print("This is epsilon pe: ", self.epsilon_pe)

        self.op_likelihood = OptimisticLikelihood(
            self.x_dim, self.epsilon_op, self.sigma, self.device
        )
        self.pe_likelihood = PessimisticLikelihood(
            self.x_dim, self.epsilon_pe, self.sigma, self.device
        )

    def forward(self, x: torch.Tensor, verbose: bool = False):
        if verbose or self.verbose:
            print(f"N_neg: {self.X_feas_neg.shape}, N_pos: {self.X_feas_pos.shape}")

        # pessimistic part
        # if self.X_feas_pos.shape[0] > 0:
        u = self.pe_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_pos.shape[0], -1]),
            self.X_feas_pos,
            verbose=self.verbose,
        )
        F_pe = self.pe_likelihood._forward(
            u, x.expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos
        )
        denom = torch.logsumexp(F_pe, -1)
        # else:
        #     denom = torch.tensor(0.0, device=self.device)

        # optimistic part
        # if self.X_feas_neg.shape[0] > 0:
        v = self.op_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_neg.shape[0], -1]),
            self.X_feas_neg,
            verbose=self.verbose,
        )
        F_op = self.op_likelihood._forward(
            v, x.expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg
        )
        numer = torch.logsumexp(-F_op, -1)
        # else:
        #     numer = torch.tensor(0.0, device=self.device)

        result = numer - denom
        return result, denom, numer


# ---------- high-level RBR generator (callable used by CARLA wrapper) ----------


def robust_bayesian_recourse(
    raw_model: Any,
    x0: np.ndarray,
    cat_features_indices: Optional[Sequence[int]] = None,
    train_data: Optional[np.ndarray] = None,
    num_samples: int = 200,
    perturb_radius: float = 0.2,
    delta_plus: float = 1.0,
    sigma: float = 1.0,
    epsilon_op: float = 0.5,
    epsilon_pe: float = 1.0,
    max_iter: int = 1000,
    dev: str = "cpu",
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:

    # helper to call raw_model.predict consistently
    def predict_fn_np(x):
        # raw_model might accept (n,d) and return probs or labels
        # preds_tensor = raw_model.predict(x)

        # # print(f"This is the pred_tensor {preds_tensor}")

        # if preds_tensor.ndim == 1:
        #     preds_tensor = preds_tensor.unsqueeze(0)

        # preds = preds_tensor.cpu().detach().numpy()
        # # print(f"The prediction is {preds} before numpy array")
        # preds = np.asarray(preds)

        # # convert to single-label 0/1 if probabilities provided
        # if preds.ndim == 2 and preds.shape[1] > 1:
        #     return preds.argmax(axis=1)
        # if preds.dtype.kind in ("f",):
        #     return (preds >= 0.5).astype(int).squeeze()
        # preds = preds.astype(int)

        # if x.ndim == 1:
        #     return preds.squeeze()
        # return preds
        # return torch.tensor(raw_model.predict(x.cpu().detach()))
        # Ensure 2D shape: (batch_size, outputs)
        preds_tensor = raw_model.predict(x)

        # Ensure 2D batch shape
        if preds_tensor.ndim == 1:
            preds_tensor = preds_tensor.unsqueeze(0)

        # Move to CPU numpy
        preds = preds_tensor.detach().cpu().numpy()

        # ---- CASE 1: Softmax output (shape: [N, 2+]) ----
        if preds.ndim == 2 and preds.shape[1] > 1:
            preds = preds.argmax(axis=1)

        # ---- CASE 2: Sigmoid or 1D probability (shape: [N]) ----
        elif preds.dtype.kind == "f":

            # If shape is Nx1 → squeeze to N
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.squeeze()

            # If probabilities → threshold
            preds = (preds >= 0.5).astype(int)

        # ---- CASE 3: Raw logits (single value per instance) ----
        elif preds.ndim == 2 and preds.shape[1] == 1:
            probs = 1 / (1 + np.exp(-preds))
            preds = (probs >= 0.5).astype(int)

        # ---- Fallback ----
        else:
            preds = preds.astype(int)

        # --- FINAL STEP: return scalar if batch size = 1 ---
        if preds.size == 1:
            #print("This is the prediction ", int(preds.item()))
            return int(preds.item())
        
        #print("This is the prediction ", preds)
        return preds

    # find boundary point between x0 and nearest opposite-label train point
    def dist(a: torch.Tensor, b: torch.Tensor):
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    # feasible set sampled around x_b
    def uniform_ball(x: torch.Tensor, r: float, n: int, rng_state):
        rng_local = check_random_state(rng_state)
        # print(f"this is x: {x}")
        d = x.shape[0]
        # print(d)
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
        rho = torch.nonzero(u * torch.arange(1, p + 1).to(dev) > (cssv - delta))[
            -1, 0
        ]
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
    # if "cuda" in device and torch.cuda.is_available():
    #     dev = torch.device(device)
    # else:
    #     dev = torch.device("cpu")

    rng = check_random_state(random_state)

    if train_data is None:
        raise ValueError("train_data must be provided to robust_bayesian_recourse")

    # ------- Implementation of fit_instance() ------------------
    x0_t = torch.from_numpy(x0.copy()).float().to(dev)
    print(f"x0_t: {x0_t}")

    train_t = torch.tensor(train_data).float().to(dev)
    # print(f"train_t: {train_t}")

    # training label vector
    train_label = torch.tensor(predict_fn_np(train_t)).to(dev)
    # print(f"train_label: {train_label}")

    # -------- Implementation of find_x_boundary() ---------------
    # find nearest opposite label examples and search along line for boundary
    x_label = torch.tensor(predict_fn_np(x0_t.clone()), device=dev)
    print(f"x_label: {x_label}")

    dists = dist(train_t, x0_t)
    order = torch.argsort(dists)
    # print(f"order: {order}")
    candidates = train_t[order[train_label[order] == (1 - x_label)]][:1000]
    # print(f"candidates: {candidates}")
    best_x_b = None
    best_dist = torch.tensor(float("inf"), device=dev)

    for x_c in candidates:
        lambdas = torch.linspace(0, 1, 100, device=dev)
        for lam in lambdas:
            x_b = (1 - lam) * x0_t + lam * x_c
            label = predict_fn_np(x_b)
            if label == 1 - x_label:
                curdist = dist(x0_t, x_b)
                if curdist < best_dist:
                    best_x_b = x_b.detach().clone()
                    best_dist = curdist.detach().clone()
                break
    # ------------------ end of find_x_boundary() --------------------

    if best_x_b is None:
        # fallback: nearest opposite neighbor directly
        opp_idx = (train_label == (1 - x_label)).nonzero(as_tuple=False)
        if opp_idx.shape[0] == 0:
            # can't find opposite label in train set -> return original
            return x0.copy()
        first_idx = opp_idx[0, 0].item()
        best_x_b = train_t[first_idx].detach().clone()
        best_dist = dist(x0_t, best_x_b)

    delta = best_dist + delta_plus

    print(f"best_x_b: {best_x_b}, delta: {delta}")

    X_feas = uniform_ball(best_x_b, perturb_radius, num_samples, rng).float().to(dev)

    # apply categorical clamping if requested
    # if cat_features_indices:
    #     for i in range(X_feas.shape[0]):
    #         X_feas[i] = reconstruct_encoding_constraints(X_feas[i], cat_features_indices)

    y_feas = predict_fn_np(X_feas)

    # print(f"X_feas: {X_feas}")
    # print(f"y_feas: {y_feas}")

    if (y_feas == 1).any():
        X_feas_pos = X_feas[y_feas == 1].reshape([int((y_feas == 1).sum().item()), -1])
    else:
        X_feas_pos = torch.empty((0, X_feas.shape[1]), device=dev)

    # print(f"X_feas_pos: {X_feas_pos}")

    if (y_feas == 0).any():
        X_feas_neg = X_feas[y_feas == 0].reshape([int((y_feas == 0).sum().item()), -1])
    else:
        X_feas_neg = torch.empty((0, X_feas.shape[1]), device=dev)

    # print(f"[Debug] X_feas_pos shape: {X_feas_pos.shape}, X_feas_neg shape: {X_feas_neg.shape}")
    # print(f"X_feas_neg: {X_feas_neg}")
    # torch.autograd.set_detect_anomaly(True) # try to catch NaNs
    # build loss wrapper
    loss_fn = RBRLoss(
        X_feas,
        X_feas_pos,
        X_feas_neg,
        epsilon_op,
        epsilon_pe,
        sigma,
        device=dev,
        verbose=verbose,
    )

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

        # print(f"x_new: {x_new}")
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
    # print(f"Final counterfactual cf: {cf}")

    # ----------------------------- end of optimize() -----------------------

    return cf
