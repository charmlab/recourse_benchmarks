import sys
# which to use? try it and see
# sys.path.append('./')
sys.path.insert(0, '.') 
import os

from models.classifiers import ann
import utils
import torch
import data.utils as dutils
from sklearn.utils.class_weight import compute_sample_weight
import argparse
import logging

def _train_ann(HIDDEN_LAYER_SIZE,DATASET_STR,CALIBRATED,LABEL_SRC,MIN_MAX,TRAIN_TEST_LABEL_SRC,FORCE_RETRAIN,SEED,N_EPOCHS,LEARNING_RATE, BATCH_SIZE,DEVICE,LABEL_PATH):
    CLF_FOLDER = utils.get_ann_folder(HIDDEN_LAYER_SIZE,DATASET_STR,LABEL_SRC,CALIBRATED,MIN_MAX,TRAIN_TEST_LABEL_SRC)
    os.makedirs(CLF_FOLDER, exist_ok=True)
    STATE_PATH = os.path.join(CLF_FOLDER, "state.pth")
    if os.path.exists(STATE_PATH) and not FORCE_RETRAIN:
        # print the hidden params of this model
        print("[INFO] trained model already exists. Do FORCE_RETRAIN=True to train again")
        train_y,train_X,test_y,test_X = dutils.load_dataset(DATASET_STR, cust_labels_path=LABEL_PATH, ret_tensor=True, min_max=MIN_MAX)
        INPUT_SHAPE = train_X.shape[1]

        LAYER_SIZE = [INPUT_SHAPE] + HIDDEN_LAYER_SIZE
        ann_model = ann.BinaryClassifier(LAYER_SIZE[:])
        state = torch.load(STATE_PATH)
        ann_model.load_state_dict(state["state_dict"])
        return ann_model
    else:
        LOG_PATH = os.path.join(CLF_FOLDER, "train.log")
        print(f"logs saved file at {os.path.abspath(LOG_PATH)}")
        logger = logging.getLogger(__name__)
        utils.configLogger(logger,
                        level=logging.DEBUG,
                        format="%(asctime)s :: %(levelname)8s :: %(filename)s:%(funcName)s :: %(message)s",
                        handlers=[
                            logging.FileHandler(LOG_PATH, mode="w"),
                            logging.StreamHandler()
                        ],
        )
        utils.set_seed(SEED)
    
        logger.info(f"seed set to {SEED}")
        train_y,train_X,test_y,test_X = dutils.load_dataset(DATASET_STR, cust_labels_path=LABEL_PATH, ret_tensor=True, min_max=MIN_MAX)
        INPUT_SHAPE = train_X.shape[1]

        LAYER_SIZE = [INPUT_SHAPE] + HIDDEN_LAYER_SIZE

        logger.info(f"Training the classifier for {N_EPOCHS} epochs")

        if CALIBRATED:
            sample_weights = compute_sample_weight(class_weight='balanced',y=train_y)
            train_w_tensor = torch.from_numpy(sample_weights)
        else:
            train_w_tensor = torch.ones_like(train_y)
        
        train_dataset = torch.utils.data.TensorDataset(train_X,train_y, train_w_tensor) 
        train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,drop_last=False)
        # TODO: should test loss also use wieght?
        test_dataset = torch.utils.data.TensorDataset(test_X,test_y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,drop_last=False)

        ann_model = ann.BinaryClassifier(LAYER_SIZE[:]).to(DEVICE)

        optimizer = torch.optim.Adam(ann_model.parameters(), lr=LEARNING_RATE)

        losslog = {"trainloss":[],"testloss":[]}
        for epoch in range(1, N_EPOCHS+1):
            train_loss = ann.train_epoch(ann_model, optimizer, train_loader, epoch,DEVICE)
            test_loss = ann.eval_epoch(ann_model, test_loader, epoch, DEVICE)
            losslog['trainloss'].append(train_loss)
            losslog['testloss'].append(test_loss)

        utils.save_curves(losslog, CLF_FOLDER)

        # prediction tensor should be constructed as a batch but let it be?
        test_predprob_tensor = ann_model(test_X.to(DEVICE)).cpu().detach()
        test_pred_tensor = ((test_predprob_tensor>0.5)*1.0).cpu().detach()
        utils.clf_eval(test_y, test_pred_tensor, test_predprob_tensor, logger)
        state = {
                'epoch': N_EPOCHS,
                'state_dict': ann_model.state_dict(),
                }
        state.update(losslog)
        torch.save(state, STATE_PATH)
        logger.info(f"saved trained model at:{STATE_PATH}")
        return ann_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train ANN model')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--dataset', type=str, help='dataset to train on')
    parser.add_argument('--force-retrain', action='store_true', help='flag to specify if model should be retrained')
    parser.add_argument('--lsrc', type=str, help='which labels should be used')
    parser.add_argument('--calibrated', action='store_true', help='flag to specify if model should be calibrated')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch', type=int, default=2048, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--device', type=int, default=0, help='device to train on')
    parser.add_argument('--hidd', nargs='+',default=[], type=int)
    parser.add_argument('--to', action='store_false', help='flag to specify if lsrc should be only trained on train split only')
    parser.add_argument('--std', action='store_false', help='flag to specify if std normalisation should be uses')
    

    args = parser.parse_args()
    SEED = args.seed
    DATASET_STR = args.dataset
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch

    # 狄外思
    # DEVICE = torch.device(f'cuda:{args.device}')
    # DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    # DEVICE = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    LEARNING_RATE = args.lr
    FORCE_RETRAIN = args.force_retrain
    LABEL_SRC = args.lsrc
    TRAIN_TEST_LABEL_SRC = args.to
    CALIBRATED = args.calibrated
    MIN_MAX = args.std
    
    # HIDDEN_LAYER_SIZE = [10,10,10] # [8,4] for unicirc and circles, [4,4] for rest, [10,10,10] for big boys
    HIDDEN_LAYER_SIZE = args.hidd
    if LABEL_SRC == 'gold':
        assert not(TRAIN_TEST_LABEL_SRC), "whether label src is trained on train test or not, applicable when src is not gold"

    if LABEL_SRC == 'xgb':
        LABEL_PATH =  utils.get_xgb_folder(DATASET_STR,TRAIN_TEST_LABEL_SRC,MIN_MAX)
    elif LABEL_SRC == 'rf':
        LABEL_PATH =  utils.get_rf_folder(DATASET_STR,TRAIN_TEST_LABEL_SRC,MIN_MAX)
    elif LABEL_SRC == 'gold':
        LABEL_PATH = None
    else: 
        raise NotImplementedError

    ann_model = _train_ann(HIDDEN_LAYER_SIZE,DATASET_STR,CALIBRATED,LABEL_SRC,MIN_MAX,TRAIN_TEST_LABEL_SRC,FORCE_RETRAIN,SEED,N_EPOCHS,LEARNING_RATE, BATCH_SIZE,DEVICE,LABEL_PATH)