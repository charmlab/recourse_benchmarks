import logging
import math
import os
import random
import signal
import subprocess
import sys
from datetime import datetime
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from methods.catalog.genre.library.models.classifiers.ann import BinaryClassifier as ann_BinaryClassifier
from matplotlib import cm
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def curr_time_hash(n=0):
    current_branch = (
        subprocess.check_output(["git", "branch", "--show-current"])
        .strip()
        .decode("utf-8")
    )
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", f"@~{n}"])
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        commit_hash = "none"
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{current_branch}@{commit_hash}-{timestamp}"


def set_handler(func):
    def signal_handler(signum, frame):
        signame = signal.Signals(signum).name
        print(f"Signal handler called with signal {signame} ({signum})")
        func()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def get_pair_dist(input1, input2, p=1):
    """
    args: B1xD input1, B2xD input2
    return B1xB2 shape matrix with pairwise distance between all rows of input1 and input 2
    """
    return torch.norm(input1[:, None, :] - input2[None, :, :], p=p, dim=-1)


def bgrad(func, x):
    """A function to compute the (batched)gradient of a scalar-valued function func with respect to x. BxD(x1) ---> BxD(df/dx1)"""
    out = func(x)
    return torch.autograd.grad(
        out, x, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]


def ret_statistics(L):
    """helper function to return useful statistics from list data"""
    npl = np.array(L)

    mean = np.mean(npl, axis=0)
    std = np.std(npl, axis=0)
    max_val = np.max(npl, axis=0)
    max_ind = np.argmax(npl, axis=0)
    return mean, std, max_val, max_ind


def msstr(data, with_std=True):
    """given list like data, returns mean and std formatted to 2 decimal places"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(data, List):
        data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if with_std:
        return f"{mean:.02f}±{std:.02f}"
    else:
        return f"{mean:.02f}"


def metrics_dict2df(metric_dict, metrics=None, column_names=["Dataset", "Metric"]):
    if metrics is None:
        metrics = list(metric_dict.keys())
    df = pd.concat(
        {
            metric: pd.DataFrame.from_dict(data)
            for metric, data in metric_dict.items()
            if metric in metrics
        },
        axis=1,
    )

    df.columns = df.columns.swaplevel(0, 1)
    df.sort_index(axis=1, level=0, inplace=True)

    # Set column names for MultiIndex
    df.columns.names = column_names
    return df


def set_seed(seed):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)


def split_data(X, y, split_ratio=0.9):
    size = len(X)
    indices = torch.randperm(size)
    split_idx = int(split_ratio * size)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def clf_eval_print(y, pred, predprob):
    # works for cpu tensors and np arrays
    matrix = confusion_matrix(y, pred)
    class_accuracy = matrix.diagonal() / matrix.sum(axis=1)

    print(f"Accuracy per class ={class_accuracy*100}")
    print("Confusion Matrix")
    print(matrix)
    print("Classification report: ")
    # classification report returns a string but can also return a dictionary
    print(classification_report(y, pred, digits=4))
    roc_auc = roc_auc_score(y, predprob)
    print(f"ROC AUC Score: {roc_auc:0.4f}")


def clf_eval(y, pred, predprob, logger):
    # works for cpu tensors and np arrays
    matrix = confusion_matrix(y, pred)
    class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
    roc_auc = roc_auc_score(y, predprob)
    logger.info(
        f"Accuracy per class: Class 0: {class_accuracy[0]*100:.4f}%, Class 1: {class_accuracy[1]*100:.4f}%"
    )
    logger.info(f"Confusion Matrix:\n {matrix}")
    # can also return a dictionary
    logger.info(
        f"Classification Report:\n {classification_report(y, pred,digits=4)} ROC AUC  {roc_auc:0.4f}"
    )


def _construct_name(initial_str: str, layer_list: list):
    """
    'ANN', [3,4,4] ---> ANN_3_4_4
    """
    newname = initial_str
    for layer in layer_list:
        newname += "_" + str(layer)
    return newname


def get_xgb_folder(
    DATASET_STR, TRAIN_ON_TEST=True, MIN_MAX=True, base_dir="./saved_models", **kwargs
):
    return os.path.abspath(
        f"{base_dir}/classifiers/{DATASET_STR}/{'xgb'}_{'tt' if TRAIN_ON_TEST else 'to'}_{'mm' if MIN_MAX else 'std'}/"
    )


def get_rf_folder(
    DATASET_STR, TRAIN_ON_TEST=True, MIN_MAX=True, base_dir="./saved_models", **kwargs
):
    return os.path.abspath(
        f"{base_dir}/classifiers/{DATASET_STR}/{'rf'}_{'tt' if TRAIN_ON_TEST else 'to'}_{'mm' if MIN_MAX else 'std'}/"
    )


def get_ann_folder(
    HIDDEN_LAYER_SIZE,
    DATASET_STR,
    LABEL_SRC,
    CALIBRATED=False,
    MIN_MAX=True,
    TRAIN_TEST_LABEL_SRC=True,
    base_dir="./saved_models",
    **kwargs,
):
    CLF_NAME = _construct_name(
        f"ann_{LABEL_SRC}_{'tt' if TRAIN_TEST_LABEL_SRC else 'to'}_{'mm' if MIN_MAX else 'std'}{'_cali' if CALIBRATED else ''}",
        HIDDEN_LAYER_SIZE,
    )
    return os.path.abspath(f"{base_dir}/classifiers/{DATASET_STR}/{CLF_NAME}")


def get_pairmodel_folder(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    NHEAD,
    EMB_SIZE,
    TOP_K,
    DATASET_STR,
    YSTAR,
    DIMFF,
    LABEL_SRC,
    TRAIN_TEST_LABEL_SRC,
    EXTRA_STR="",
    MIN_MAX=True,
    base_dir="./saved_models",
    **kwargs,
):
    PM_NAME = f"{LABEL_SRC}_{'tt' if TRAIN_TEST_LABEL_SRC else 'to'}_{'mm' if MIN_MAX else 'std'}_ystar{YSTAR}_{EXTRA_STR}_e{NUM_ENCODER_LAYERS}_d{NUM_DECODER_LAYERS}_h{NHEAD}_emb{EMB_SIZE}_k{TOP_K}_ff{DIMFF}"
    return os.path.abspath(f"{base_dir}/pairmodels/{DATASET_STR}/{PM_NAME}")


def load_ann(INPUT_SHAPE, HIDDEN_LAYER_SIZE, DATASET_STR, **kwargs):
    LAYER_SIZE = [INPUT_SHAPE] + HIDDEN_LAYER_SIZE

    CLF_FOLDER = get_ann_folder(HIDDEN_LAYER_SIZE, DATASET_STR, **kwargs)
    STATE_PATH = f"{CLF_FOLDER}/state.pth"
    classifier_m = ann_BinaryClassifier(LAYER_SIZE)
    state = torch.load(STATE_PATH, map_location=torch.device("cpu"))
    classifier_m.load_state_dict(state["state_dict"])
    print(f"[INFO] loaded ann model from {os.path.abspath(CLF_FOLDER)}")
    return classifier_m, os.path.abspath(CLF_FOLDER)


def plot_curves(ax, loss_dict, _slice=None):
    for name, losslist in loss_dict.items():
        if len(losslist) != 0:
            filtered_losslist = [
                (i, loss) for i, loss in enumerate(losslist) if loss is not None
            ]
            if len(filtered_losslist) > 0:
                x_vals, y_vals = zip(*filtered_losslist)

                if _slice is None:
                    ax.plot(x_vals, y_vals, label=name)
                else:
                    sliced_x_vals = [x for x in x_vals if _slice[0] <= x < _slice[1]]
                    sliced_y_vals = [
                        y for i, y in zip(x_vals, y_vals) if _slice[0] <= i < _slice[1]
                    ]
                    ax.plot(sliced_x_vals, sliced_y_vals, label=name)

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid()


def save_curves(loss_dict, save_dir, _name="loss_curves"):
    fig, ax = plt.subplots()
    plot_curves(ax, loss_dict)
    plt.savefig(f"{save_dir}/{_name}.png")
    plt.clf()


def configLogger(logger, level, format, handlers):
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, save_dir, name):
    save_path = os.path.join(save_dir, name + ".yaml")
    with open(save_path, "w") as file:
        yaml.dump(config, file)


def prepare_args(parser):
    args = parser.parse_args()
    config = load_config(args.config) if args.config != None else {}
    if config == None:
        config = {}
    for key, value in vars(args).items():
        if value == None:
            try:
                vars(args)[key] = config[key]  # hyphen key can cause a problem.
            except KeyError:
                if key != "config":
                    sys.exit(f"{key} is not passed as arg, nor in config file")
                if key != "config":
                    assert vars(args)[key] != None
    return args


# https://www.color-hex.com/
pos_scatter_clr = "#79B9E1"  # dark skyblue
neg_scatter_clr = "lightcoral"


def scatter2d(ax, data, label=None, data_name="data", show_axis_name=True, **kwargs):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if label is None:
        ax.scatter(data[:, 0], data[:, 1], label=data_name, **kwargs)
    else:
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        data_0 = data[label == 0]
        data_1 = data[label == 1]

        ax.scatter(
            data_0[:, 0],
            data_0[:, 1],
            label=f"{data_name} - class 0",
            **kwargs,
            color=neg_scatter_clr,
        )
        ax.scatter(
            data_1[:, 0],
            data_1[:, 1],
            label=f"{data_name} - class 1",
            **kwargs,
            color=pos_scatter_clr,
        )
    if show_axis_name:
        ax.set_ylabel(r"$x_2$")
        ax.set_xlabel(r"$x_1$")
    # ax.legend()


# https://www.color-hex.com/
pos_surface_clr = "LightSkyBlue"
neg_surface_clr = "#f4a6a6"
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html
@torch.no_grad()
def viz_clf(
    ax,
    model,
    data=None,
    label=None,
    DEVICE=torch.device("cpu"),
    hard=False,
    bound_x=(0, 1),
    bound_y=(0, 1),
    step=0.001,
    pad=0.05,
    auto_bound=True,
    contour=False,
    **kwargs,
):

    if data is not None:
        scatter2d(ax, data, label, **kwargs)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    if auto_bound:
        bound_x = ax.get_xlim()
        bound_y = ax.get_ylim()
        bound_sq = (min(bound_x[0], bound_y[0]), max(bound_x[1], bound_y[1]))
        bound_x = bound_sq
        bound_y = bound_sq
    # pad the bounds for better plots
    bound_x = (bound_x[0] - pad, bound_x[1] + pad)
    bound_y = (bound_y[0] - pad, bound_y[1] + pad)

    c0 = torch.Tensor(to_rgba(neg_surface_clr)).to(DEVICE)
    c1 = torch.Tensor(to_rgba(pos_surface_clr)).to(DEVICE)
    x1 = torch.arange(*bound_x, step=step, device=DEVICE)
    x2 = torch.arange(*bound_y, step=step, device=DEVICE)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing="xy")  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    leng = model_inputs.shape[0]
    width = model_inputs.shape[1]

    # checks if the model is an instance of sklearn classifiers, not ideal
    sk_clf = "sklearn" in str(model.__class__)

    if sk_clf:
        if hard:  # hax
            preds = torch.from_numpy(
                model.predict(model_inputs.reshape(-1, 2)).reshape(leng, width, -1)
            )
        else:
            preds = torch.from_numpy(
                model.predict_proba(model_inputs.reshape(-1, 2))[:, 1].reshape(
                    leng, width, -1
                )
            )
    else:
        model.to(DEVICE)
        model.eval()
        if hard:
            preds = 1.0 * (model(model_inputs) > 0.5)
        else:
            preds = model(model_inputs)

    if contour:
        if isinstance(xx1, torch.Tensor):
            xx1 = xx1.cpu().numpy()
            xx2 = xx2.cpu().numpy()
            preds = preds.cpu().numpy()
            preds = preds.squeeze()

        ax.contour(xx1, xx2, preds, levels=[0.5], colors="k", linestyles="--", **kwargs)
        ax.set_xlim(bound_x)
        ax.set_ylim(bound_y)
    else:
        output_image = (1 - preds) * c0[None, None] + preds * c1[
            None, None
        ]  # Specifying "None" in a dimension creates a new one
        if isinstance(output_image, torch.Tensor):
            output_image = output_image.cpu().numpy()
        ax.imshow(output_image, origin="lower", extent=(*bound_x, *bound_y), alpha=0.6)


pos_pair_clr = "deepskyblue"
neg_pair_clr = "tomato"


def plot_pairs(ax, src, tgt, colors=None, arrows=True, names=None, **kwargs):
    if names is None:
        names = ("src", "tgt")
    scatter2d(ax, src, data_name=names[0], color=neg_pair_clr, **kwargs)
    scatter2d(ax, tgt, data_name=names[1], color=pos_pair_clr, **kwargs)

    if arrows:
        if isinstance(src, torch.Tensor):
            src = src.cpu().numpy()
        if isinstance(tgt, torch.Tensor):
            tgt = tgt.cpu().numpy()

        # colors = np.where(color, 'green', 'red')
        direc = tgt - src
        if colors is None:
            ax.quiver(
                src[:, 0],
                src[:, 1],
                direc[:, 0],
                direc[:, 1],
                scale_units="xy",
                angles="xy",
                scale=1,
                width=0.0025,
                color="black",
            )
        else:
            ax.quiver(
                src[:, 0],
                src[:, 1],
                direc[:, 0],
                direc[:, 1],
                colors,
                scale_units="xy",
                angles="xy",
                scale=1,
                width=0.0025,
                cmap=mcolors.ListedColormap([pos_pair_clr, neg_pair_clr]),
                linewidth=2,
            )


def eval_callbacks(callbacks, x, device=None):
    evaluated_hist = {}
    for name, func in callbacks.items():
        evaluated_hist[name] = []
        for i, a in enumerate(x):
            if device is not None:
                a = a.to(device)
            evaluated_hist[name].append(func(a).mean().item())
    return evaluated_hist


def sample(*args, size):
    leng_ = len(args[0])
    for x in args[1:]:
        leng = len(x)
        assert leng_ == leng

    if size > leng_:
        return args
    else:
        ind = np.random.choice(leng_, size, replace=False)

        sampled_args = ()
        for x in args:
            sampled_args = sampled_args + (x[ind],)
        if len(args) > 1:
            return sampled_args
        else:
            return sampled_args[0]


@torch.no_grad()
def viz_pm(
    xf_r,
    YSTAR,
    DEVICE,
    pair_model,
    data_X=None,
    data_y=None,
    bound_x=(0, 1),
    bound_y=(0, 1),
    step=0.01,
    cols=3,
):
    num = len(xf_r)
    if num % cols != 0:
        print(f"would be best if number of examples is multiple of cols={cols} ")
    xf_r = xf_r.to(DEVICE)
    pair_model = pair_model.to(DEVICE)

    pad = step
    bound_x = (bound_x[0] - pad, bound_x[1] + pad)
    bound_y = (bound_y[0] - pad, bound_y[1] + pad)
    x1 = torch.arange(*bound_x, step=step, device=DEVICE)
    x2 = torch.arange(*bound_y, step=step, device=DEVICE)

    xx1, xx2 = torch.meshgrid(x1, x2, indexing="xy")  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)

    # utils.set_seed(21)
    response = torch.zeros(xf_r.shape[:1] + xx1.shape)
    pair_model.eval()

    def conditional_lklh(xcf, xf, yf, ycf):
        yf = yf.view(-1, 1)
        ycf = ycf.view(-1, 1)
        outp = pair_model(xf, yf, xcf, ycf)
        return ((xcf - outp) ** 2).sum(dim=1, keepdim=True)

    xcf = model_inputs.reshape(-1, 2)  # reshape grid to tall tensor
    batch_size = xcf.shape[0]  # gets the batch size
    for i in range(num):
        # this loop can be eliminated as well but hey, it's ok
        xf = (
            xf_r[i].unsqueeze(0).repeat(batch_size, 1)
        )  # selects example i and expands it to match size of xcf
        ycf = torch.ones((batch_size), device=DEVICE) * YSTAR
        yf = 1 - ycf
        response[i] = conditional_lklh(xcf, xf, yf, ycf).reshape(xx1.shape)

    fig, ax = plt.subplots(
        math.ceil(num / cols), cols, figsize=(30, 8.5 * (math.ceil(num / cols)))
    )
    ax = ax.flatten()
    for i in range(num):
        X = xx1.detach().cpu().numpy()
        Y = xx2.detach().cpu().numpy()
        Z = response[i].detach().cpu().numpy()

        ax[i].contourf(X, Y, Z, 50, cmap=cm.coolwarm)
        if data_X != None:
            scatter2d(ax[i], data_X, data_y)
        # plot particular point
        ax[i].plot(xf_r.cpu()[i, 0], xf_r.cpu()[i, 1], "ro")
        ax[i].legend()
    return fig


def get_denormaliser(norm_consts):
    func = (
        lambda x: (
            ((x * norm_consts["range"]) + norm_consts["min"]) * norm_consts["std"]
        )
        + norm_consts["mean"]
    )
    return func


def get_names_hp(base_str, hyper_list):
    # hyper list is a list of tuples [('h1',values_h1),('h2', values_h2)]]
    # get names along with hparams
    ll = [base_str]
    for name, hll in hyper_list:
        local_ll = []
        for hval in hll:
            local_ll += [ele + "_" + name + str(hval) for ele in ll]
        ll = local_ll
    return ll
