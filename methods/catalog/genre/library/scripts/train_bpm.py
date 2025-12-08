import sys

# sys.path.append("./")
sys.path.insert(0, ".")

import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import utils

import data.utils as dutils
import models.binnedpm as bpm
from data.pair_data import StochasticPairsImmut

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to train pair model")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="dataset(s) to run on",
        type=str,
    )
    # parser.add_argument('--ystar', type=int,required=True, help='desired label')
    parser.add_argument("--device", type=int, required=True, help="device to train on")
    parser.add_argument("--tlamb", nargs="+", required=True, type=float)
    parser.add_argument("--perc", nargs="+", required=True, type=float)
    args = parser.parse_args()

    exp_config = utils.load_config("results/exp1_config.yaml")
    train_lamblist = args.tlamb
    gamma = 0.7
    for DATASET_STR in args.datasets:
        for train_lambda in train_lamblist:
            print(
                f"----------------------------- Executing for Dataset {DATASET_STR} -----------------------------"
            )
            YSTAR = 1.0

            # 狄外思
            # DEVICE = f'cuda:{args.device}'
            # DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # DEVICE = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            DEVICE = torch.device(
                f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )
            # DEVICE = torch.device("cpu")

            MIN_MAX = True
            TOP_K = 1000
            TRAIN_TEST_LABEL_SRC = True
            TRAIN_P = 1
            SEED = 42

            utils.set_seed(SEED)

            # load dataset
            (
                train_y,
                train_X,
                test_y,
                test_X,
                cat_mask,
                immutable_mask,
            ) = dutils.load_dataset(
                DATASET_STR, ret_tensor=True, min_max=True, ret_masks=True
            )
            INPUT_SHAPE = train_X.shape[1]

            LABEL_PATH = utils.get_rf_folder(DATASET_STR, TRAIN_TEST_LABEL_SRC, MIN_MAX)
            train_y, train_X, test_y, test_X = dutils.load_dataset(
                DATASET_STR,
                cust_labels_path=LABEL_PATH,
                ret_tensor=True,
                min_max=MIN_MAX,
            )

            INPUT_SHAPE = train_X.shape[1]

            ann_clf, ann_folder = utils.load_ann(
                INPUT_SHAPE=INPUT_SHAPE,
                DATASET_STR=DATASET_STR,
                LABEL_SRC="rf",
                **exp_config["common"],
                **exp_config["ann"][DATASET_STR],
            )

            train_pred = ((ann_clf(train_X) > gamma) * 1.0).cpu().detach().squeeze()
            test_pred = ((ann_clf(test_X) > gamma) * 1.0).cpu().detach().squeeze()

            train_Dsrc_X = train_X[train_pred == (1 - YSTAR)]
            train_Dtgt_X = train_X[(train_pred == YSTAR) & (train_y == YSTAR)]
            train_Dsrc_y = train_y[train_pred == (1 - YSTAR)]
            train_Dtgt_y = train_y[(train_pred == YSTAR) & (train_y == YSTAR)]

            print("target: ", len(train_Dtgt_X), "src", len(train_Dsrc_X))
            # paired_data = StochasticPairs(train_Dsrc_X,train_Dtgt_X,train_Dsrc_y,train_Dtgt_y,immutable_mask,lambda_=train_lambda,k=TOP_K,p=TRAIN_P)

            # pair_data_train, pair_data_val = torch.utils.data.random_split(paired_data, [int(0.9*len(paired_data)), len(paired_data) - int(0.9*len(paired_data))], generator=torch.Generator().manual_seed(SEED))

            train_Dsrc_X, train_Dsrc_y, val_Dsrc_X, val_Dsrc_y = utils.split_data(
                train_Dsrc_X, train_Dsrc_y, 0.9
            )
            train_Dtgt_X, train_Dtgt_y, val_Dtgt_X, val_Dtgt_y = utils.split_data(
                train_Dtgt_X, train_Dtgt_y, 0.9
            )
            paired_data_train = StochasticPairsImmut(
                train_Dsrc_X,
                train_Dtgt_X,
                train_Dsrc_y,
                train_Dtgt_y,
                immutable_mask,
                lambda_=train_lambda,
                k=TOP_K,
                p=TRAIN_P,
            )
            paired_data_val = StochasticPairsImmut(
                val_Dsrc_X,
                val_Dtgt_X,
                val_Dsrc_y,
                val_Dtgt_y,
                immutable_mask,
                lambda_=train_lambda,
                k=TOP_K,
                p=TRAIN_P,
            )

            test_Dsrc_X = test_X[test_pred == (1 - YSTAR)]
            test_Dtgt_X = test_X[(test_pred == YSTAR) & (test_y == YSTAR)]
            test_Dsrc_y = test_y[test_pred == (1 - YSTAR)]
            test_Dtgt_y = test_y[(test_pred == YSTAR) & (test_y == YSTAR)]
            paired_data_test = StochasticPairsImmut(
                test_Dsrc_X,
                test_Dtgt_X,
                test_Dsrc_y,
                test_Dtgt_y,
                immutable_mask,
                lambda_=train_lambda,
                k=TOP_K,
                p=TRAIN_P,
            )

            # heloc, compas-all, adult-all
            pair_model = bpm.PairedTransformerBinned(
                n_bins=50,
                num_inputs=INPUT_SHAPE,
                num_labels=1,
                num_encoder_layers=16,
                num_decoder_layers=16,
                emb_size=32,
                nhead=8,
                dim_feedforward=32,
                dropout=0.1,
            ).to(DEVICE)

            for p in pair_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            best_val_loss = float("inf")
            best_state = None
            best_epoch = -1

            num_epochs = 2000  # 2000
            # batch_size= 8192 * 2 if DATASET_STR=='adult-all' else 2048
            # batch_size= 2048 if DATASET_STR=='adult-all' else 2048
            batch_size = 1024 if DATASET_STR == "adult-all" else 1024
            print("bs: ", batch_size, "num epochs: ", num_epochs)
            learning_rate = 0.0001
            eval_freq = 10

            optimizer = torch.optim.Adam(pair_model.parameters(), lr=learning_rate)

            pair_loader_train = torch.utils.data.DataLoader(
                paired_data_train, batch_size=batch_size, drop_last=False
            )
            pair_loader_val = torch.utils.data.DataLoader(
                paired_data_val, batch_size=batch_size, drop_last=False
            )
            pair_loader_test = torch.utils.data.DataLoader(
                paired_data_test, batch_size=batch_size, drop_last=False
            )

            loss_log = {"train": [], "val": [], "test": []}

            stt = time.time()
            try:
                for batch_data in pair_loader_train:
                    print(f"[DEBUG] batch_data type: {type(batch_data)}")
                    print(f"[DEBUG] batch_data: {batch_data}")
                    if isinstance(batch_data, (list, tuple)):
                        print(f"[DEBUG] Length: {len(batch_data)}")
                        for i, item in enumerate(batch_data):
                            print(
                                f"[DEBUG] Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}"
                            )
                    break
                for epoch in range(num_epochs):
                    train_loss = bpm.train_epoch(
                        pair_model,
                        optimizer,
                        pair_loader_train,
                        epoch,
                        DEVICE,
                        show_bar=epoch % 200 == 0,
                    )
                    loss_log["train"].append(train_loss)
                    if epoch % eval_freq == 0:
                        val_loss = bpm.eval_epoch(
                            pair_model, pair_loader_val, epoch, DEVICE
                        )
                        test_loss = bpm.eval_epoch(
                            pair_model, pair_loader_test, epoch, DEVICE
                        )
                        loss_log["val"].append(val_loss)
                        loss_log["test"].append(test_loss)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state = {
                                "epoch": epoch,
                                "state_dict": pair_model.state_dict(),
                            }
                            best_epoch = epoch

                    else:
                        loss_log["val"].append(None)
                        loss_log["test"].append(None)

            except KeyboardInterrupt:
                print(f"[INFO] halted training at {epoch}")
            finn = time.time()
            print(DATASET_STR, finn - stt)
            assert best_state is not None
            best_state.update(loss_log)

            PM_OUTDIR = f"./saved_models/genre/{DATASET_STR}_gamma{gamma}/"
            PM_STATE_PATH = f"{PM_OUTDIR}/state_{train_lambda}.pth"
            os.makedirs(PM_OUTDIR, exist_ok=True)
            torch.save(best_state, PM_STATE_PATH)

            utils.save_curves(loss_log, PM_OUTDIR)
