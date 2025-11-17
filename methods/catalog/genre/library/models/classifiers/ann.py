import torch
from torch import optim, nn, utils
import torch.nn.functional as F
from tqdm import tqdm

class BinaryClassifier(nn.Module):
    def __init__(self, layer_sizes):
        super(BinaryClassifier, self).__init__()
        modules = nn.ModuleList()
        layer_sizes.append(1) # Binary Classifier
        for ch1, ch2 in zip(layer_sizes[:-1], layer_sizes[1:]):
            modules.append(nn.Linear(ch1, ch2))
            modules.append(nn.ReLU())

        modules[-1] = nn.Sigmoid()
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def _logit(self, x):
        return self.model[:-1](x)
    
def train_epoch(model, optimizer, data_loader, epoch, DEVICE):
    model.train()
    losses = []
    
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Training Epoch {epoch:03d}")
            x, y, w = batch
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.binary_cross_entropy(pred.squeeze(), y.float(),weight=w)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tepoch.set_postfix(loss=sum(losses) / len(losses))
    return sum(losses) / len(losses)

@torch.no_grad()
def eval_epoch(model, data_loader, epoch, DEVICE):
    model.eval()
    losses = []
    with tqdm(data_loader, unit="batch", leave=False) as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Evaluating Epoch {epoch:03d}")
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE) 
            pred = model(x)
            loss = F.binary_cross_entropy(pred.squeeze(), y.float())
            losses.append(loss.item())
            tepoch.set_postfix(loss=sum(losses) / len(losses))
    return sum(losses) / len(losses)