import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.distributions.normal as normal_distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
from torch.autograd import Variable

from methods.processing import reconstruct_encoding_constraints

""" 
This file contains the implementation of the Probe method, along with required helper functions 
"""

DECISION_THRESHOLD = 0.5

# Mean and variance for rectified normal distribution:
# see in here : http://journal-sfds.fr/article/view/669


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad
    grad = gradient(output, inputs)
    return grad


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.tensor(1)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_invalidation_rate_closed(torch_model, x, sigma2):
    # Compute input into CDF
    prob = torch_model(x)
    logit_x = torch.log(prob[0][1] / prob[0][0])
    Sigma2 = sigma2 * torch.eye(x.shape[0])
    jacobian_x = compute_jacobian(x, logit_x).reshape(-1)
    denom = torch.sqrt(sigma2) * torch.norm(jacobian_x, 2)
    arg = logit_x / denom
    
    # Evaluate Gaussian cdf
    normal = normal_distribution.Normal(loc=0.0, scale=1.0)
    normal_cdf = normal.cdf(arg)
    
    # Get invalidation rate
    ir = 1 - normal_cdf
    
    return ir


def perturb_sample(x, n_samples, sigma2):
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    # sample normal distributed values
    Sigma = torch.eye(x.shape[1]) * sigma2
    eps = MultivariateNormal(
        loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
    ).sample((n_samples,))
    
    return X + eps

def reparametrization_trick(mu, sigma2, n_samples):
    #var = torch.eye(mu.shape[1]) * sigma2
    std = torch.sqrt(sigma2)
    epsilon = MultivariateNormal(loc=torch.zeros(mu.shape[1]), covariance_matrix=torch.eye(mu.shape[1]))
    epsilon = epsilon.sample((n_samples,))  # standard Gaussian random noise
    ones = torch.ones_like(epsilon)
    random_samples = mu.reshape(-1) * ones + std * epsilon
    
    return random_samples


def compute_invalidation_rate(torch_model, random_samples):
    yhat = torch_model(random_samples)[:, 1]
    hat = (yhat > 0.5).float()
    ir = 1 - torch.mean(hat, 0)
    return ir


def probe_recourse(
    torch_model,
    x: np.ndarray,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.07,
    lambda_param: float = 5,
    y_target: List[int] = [0.45, 0.55],
    n_iter: int = 500,
    t_max_min: float = 1.0,
    norm: int = 1,
    clamp: bool = False,
    loss_type: str = "MSE",
    invalidation_target: float = 0.45,
    inval_target_eps: float = 0.005,
    noise_variance: float = 0.01
) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x

    Parameters
    ----------
    torch_model: black-box-model to discover
    x: factual to explain
    cat_feature_indices: list of positions of categorical features in x
    binary_cat_features: If true, the encoding of x is done by drop_if_binary
    feature_costs: List with costs per feature
    lr: learning rate for gradient descent
    lambda_param: weight factor for feature_cost
    y_target: List of one-hot-encoded target class
    n_iter: maximum number of iteration
    t_max_min: maximum time of search
    norm: L-norm to calculate cost
    clamp: If true, feature values will be clamped to (0, 1)
    loss_type: String for loss function (MSE or BCE)
    Invalidation_target: target invalidation rate
    inval_target_eps: epsilon for invalidation rate
    noise_variance: variance of the normal distribution for sampling

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cpu" # for simplicity and to avoid Runtime error.
    # returns counterfactual instance
    torch.manual_seed(0)
    noise_variance = torch.tensor(noise_variance)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float().to(device)

    #print("x:", x)

    x = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)
    # x_new is used for gradient search in optimizing process
    x_new = Variable(x.clone(), requires_grad=True)
    # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
    # such that categorical data is either 0 or 1
    
    # x_new_enc = reconstruct_encoding_constraints( #TODO: check if this is needed here, i believe that the encoding is done in the model prediction
    #     x_new, cat_feature_indices, binary_cat_features
    # )

    optimizer = optim.Adam([x_new], lr, amsgrad=True)
    softmax = nn.Softmax()

    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
        f_x_new = softmax(torch_model(x_new))[1]
    else:
        loss_fn = torch.nn.BCELoss()
        f_x_new = torch_model(x_new)[:, 1]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    costs = []
    ces = []

    random_samples = reparametrization_trick(x_new, noise_variance, n_samples=1000)
    invalidation_rate = compute_invalidation_rate(torch_model, random_samples)
    
    while (f_x_new <= DECISION_THRESHOLD) or (invalidation_rate > invalidation_target + inval_target_eps):
        # it = 0
        for it in range(n_iter):
        # while invalidation_target >= 0.5 and it < n_iter:
            
            optimizer.zero_grad()
            # x_new_enc = reconstruct_encoding_constraints(
            #     x_new, cat_feature_indices, binary_cat_features
            # )
            # use x_new_enc for prediction results to ensure constraints
            # f_x_new = softmax(torch_model(x_new))[:, 1]
            f_x_new_binary = torch_model(x_new).squeeze(axis=0)

            cost = (
                torch.dist(x_new, x, norm)
                if feature_costs is None
                else torch.norm(feature_costs * (x_new - x), norm)
            )
            
            # Compute Invalidation loss
            # output_mean, output_std = compute_output_dist_suff_statistics(torch_model, x_new,
            #                                                              noise_variance=noise_variance)
            
            # normal = normal_distribution.Normal(loc=0.0, scale=1.0)
            # ratio = torch.divide(output_mean, output_std)
            # normal_cdf = normal.cdf(ratio)
            # invalidation_rate = 1 - normal_cdf

            # invalidation_rate = compute_invalidation_rate(torch_model, random_samples)
            invalidation_rate_c = compute_invalidation_rate_closed(torch_model, x_new, noise_variance)
            
            # Compute & update losses
            loss_invalidation = invalidation_rate_c - invalidation_target
            # Hinge loss
            loss_invalidation[loss_invalidation < 0] = 0

            loss = 3 * loss_invalidation + loss_fn(f_x_new_binary, y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            random_samples = reparametrization_trick(x_new, noise_variance, n_samples=10000)
            invalidation_rate = compute_invalidation_rate(torch_model, random_samples)

            # x_pertub = perturb_sample(x_new, sigma2=noise_variance, n_samples=10000)
            # pred = 1 - torch_model(x_pertub)[:, 1]
            # invalidation_rate_empirical = torch.mean(pred)

            # print('-----------------------------------------')
            # print('IR empirical', invalidation_rate_empirical)
            # print('IR from loss', invalidation_rate)
            # print('IR loss', loss_invalidation)
            
            # clamp potential CF
            if clamp:
                x_new.clone().clamp_(0, 1)
            # it += 1
            
            # x_new_enc = reconstruct_encoding_constraints(
            #     x_new, cat_feature_indices, binary_cat_features
            # )
            # f_x_new = torch_model(x_new_enc)[:, 1]
            f_x_new = torch_model(x_new)[:, 1]

        if (f_x_new > DECISION_THRESHOLD) and (invalidation_rate < invalidation_target + inval_target_eps):
                print('--------------------------------------')
                print('invalidation rate:', invalidation_rate)
                # print('emp invalidation rate', invalidation_rate_empirical)
                print('cost:', cost)
                print('classifier output:', f_x_new_binary)
                
                costs.append(cost)
                ces.append(x_new)
                
                break
                
        lamb -= 0.10
        
        if datetime.datetime.now() - t0 > t_max:
            print("Timeout")
            break

    if not ces:
        print("No Counterfactual Explanation Found at that Target Rate - Try Different Target")
        return x_new.cpu().detach().numpy().squeeze(axis=0)
    else:
        print("Counterfactual Explanation Found")
        costs = torch.tensor(costs)
        min_idx = int(torch.argmin(costs).numpy())
        x_new_enc = ces[min_idx]
    
    #print("x_prime ", x_new_enc.cpu().detach().numpy().squeeze(axis=0))

    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
