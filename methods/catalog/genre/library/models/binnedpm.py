import sys

sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer
from tqdm import tqdm

from methods.catalog.genre.library.models.utils import (
    Embedding2RealNet,
    HetLinearLayer,
    LearnedPositionEncoding,
    PositionalEncoding,
    RealEmbeddingNet,
)


class PairedTransformerBinnedExpanded(nn.Module):
    def __init__(
        self,
        n_bins: int,
        num_inputs: int,
        num_labels: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(PairedTransformerBinnedExpanded, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # batch_first=True
        )
        # self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.real_embed = RealEmbeddingNet(
            num_inputs + num_labels, emb_size, src_dim=1 + n_bins
        )
        # self.real_embed_tgt = RealEmbeddingNet(num_inputs + num_labels, emb_size, src_dim=1+ n_bins)

        self.positional_encoding = LearnedPositionEncoding(
            emb_size, dropout=dropout, max_len=num_inputs + num_labels
        )
        causal_mask = self.generate_square_subsequent_mask(num_inputs + num_labels)
        self.register_buffer("causal_mask", causal_mask)

        self.num_labels = num_labels
        self.num_inputs = num_inputs

        # binning
        self.n_bins = (
            n_bins  # TODO: Not ideal but same number of bins for each position
        )
        self.emb2out = HetLinearLayer(emb_size, n_bins, num_inputs + num_labels)

        bins = torch.linspace(0, 1, n_bins + 1)[:-1]
        self.register_buffer("bins", bins)

    def forward(self, xf: Tensor, yf: Tensor, xcf: Tensor, ycf: Tensor):

        yf = 2 * yf - 1
        ycf = 2 * ycf - 1

        bin_weights_encoder = (
            -(self.n_bins**2) * ((xf[:, :, None] - self.bins.reshape(1, -1)) ** 2) / 2
        ).softmax(dim=-1)
        src = torch.cat((xf[:, :, None], bin_weights_encoder), dim=-1)
        src = torch.cat((yf[:, :, None].repeat(1, 1, 1 + self.n_bins), src), dim=1)

        bin_weights = (
            -(self.n_bins**2)
            * ((xcf[:, :, None] - self.bins.reshape(1, -1)) ** 2)
            / 2
        ).softmax(dim=-1)
        tgt = torch.cat((xcf[:, :, None], bin_weights), dim=-1)
        tgt = torch.cat((ycf[:, :, None].repeat(1, 1, 1 + self.n_bins), tgt), dim=1)

        src_emb = self.positional_encoding(self.real_embed(src))
        tgt_emb = self.positional_encoding(self.real_embed(tgt))
        outs_emb = self.transformer(
            src_emb, tgt_emb, tgt_mask=self.causal_mask, tgt_is_causal=True
        )
        outs = self.emb2out(outs_emb)[:, :-1]

        # -------------------- Approximate upper bound on exact loss --------------------
        log_prob = outs.log_softmax(dim=-1)
        # sanity check: exp(log_prob).sum(dim=-1) == 1 (all, BxD), bin_weights.sum(dim=-1) == 1 (all, BxD)

        loss = -(log_prob * bin_weights).sum(
            dim=-1
        )  # loss on each feature, each examples in the batch
        return loss.sum(dim=-1)  # sum along features, gives loss for each input
        # -------------------- Approximate upper bound on exact loss --------------------

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _decode(self, hist, tgt_mask, y, memory, probs):
        y = 2 * y - 1
        tgt = torch.cat((hist[:, :, None], probs), dim=-1)
        tgt = torch.cat((y[:, :, None].repeat(1, 1, 1 + self.n_bins), tgt), dim=1)

        tgt_emb = self.positional_encoding(self.real_embed(tgt))
        # just returns the logit values of bins in the next feature
        return self.emb2out(self.transformer.decoder(tgt_emb, memory, tgt_mask))[:, -1]

    def _encode(self, xf, yf):
        yf = 2 * yf - 1
        bin_weights_encoder = (
            -(self.n_bins**2) * ((xf[:, :, None] - self.bins.reshape(1, -1)) ** 2) / 2
        ).softmax(dim=-1)
        src = torch.cat((xf[:, :, None], bin_weights_encoder), dim=-1)
        src = torch.cat((yf[:, :, None].repeat(1, 1, 1 + self.n_bins), src), dim=1)
        return self.transformer.encoder(self.positional_encoding(self.real_embed(src)))

    @torch.no_grad()
    def _sample(self, xf, yf, y, sigma=0.0, temp=1.0):

        self.eval()
        yf = yf.view(-1, 1)
        y = y.view(-1, 1)
        memory = self._encode(xf, yf)  # do this once
        batch_size = xf.shape[0]
        hist = torch.zeros(
            (batch_size, 0), device=xf.device
        )  # initialize empty features, will be concatenated with y
        probs_hist = torch.zeros(
            (batch_size, 0, self.n_bins), device=xf.device
        )  # initialize empty features, will be concatenated with y

        for i in range(self.num_inputs):
            mask = self.generate_square_subsequent_mask(i + 1)
            logits = temp * self._decode(
                hist, mask, y, memory, probs_hist
            )  # gives unnormalised log probability
            probs = logits.softmax(dim=-1)  # get probability

            sampled_idx = torch.multinomial(probs, 1)  # get bin index
            bin_val = self.bins[sampled_idx]  # get bin value for each batch element
            samp_val = bin_val + sigma * torch.randn_like(
                bin_val
            )  # add sizzle to bin value
            hist = torch.cat((hist, samp_val), dim=1)
            probs_hist = torch.cat((probs_hist, probs.unsqueeze(1)), dim=1)
        return hist


class PairedTransformerBinned(nn.Module):
    def __init__(
        self,
        n_bins: int,
        num_inputs: int,
        num_labels: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(PairedTransformerBinned, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # batch_first=True
        )
        # self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.real_embed = RealEmbeddingNet(num_inputs + num_labels, emb_size)

        self.positional_encoding = LearnedPositionEncoding(
            emb_size, dropout=dropout, max_len=num_inputs + num_labels
        )
        causal_mask = self.generate_square_subsequent_mask(num_inputs + num_labels)
        self.register_buffer("causal_mask", causal_mask)

        self.num_labels = num_labels
        self.num_inputs = num_inputs

        # binning
        self.n_bins = (
            n_bins  # TODO: Not ideal but same number of bins for each position
        )
        self.fc = torch.nn.Linear(
            emb_size, n_bins
        )  # TODO: every position can have different layer as well
        bins = torch.linspace(0, 1, n_bins + 1)[:-1]
        self.register_buffer("bins", bins)

    def forward(self, xf: Tensor, yf: Tensor, xcf: Tensor, ycf: Tensor):

        yf = 2 * yf - 1
        ycf = 2 * ycf - 1
        src = torch.cat((yf, xf), dim=1)
        tgt = torch.cat((ycf, xcf), dim=1)
        src_emb = self.positional_encoding(self.real_embed(src))
        tgt_emb = self.positional_encoding(
            self.real_embed(tgt)
        )  # [ycf,xcf] --> [xcf_preds]
        # outs_emb = self.transformer(src_emb, tgt_emb, tgt_mask = self.causal_mask, tgt_is_causal = True) # IMP: no masking in enccoder for now?
        # transpose (seq_len, batch_size, emb_size)
        src_emb = src_emb.transpose(0, 1)  # [B, S, E] -> [S, B, E]
        tgt_emb = tgt_emb.transpose(0, 1)  # [B, S, E] -> [S, B, E]

        # print(f"[DEBUG] src_emb: {src_emb.shape}, tgt_emb: {tgt_emb.shape}, mask: {self.causal_mask.shape}")
        outs_emb = self.transformer(src_emb, tgt_emb, tgt_mask=self.causal_mask)

        # Transpose output back!
        outs_emb = outs_emb.transpose(0, 1)  # [S, B, E] -> [B, S, E]
        outs = self.fc(outs_emb)[:, :-1]

        log_prob = outs.log_softmax(dim=-1)
        bin_weights = (
            -(self.n_bins**2)
            * ((xcf[:, :, None] - self.bins.reshape(1, -1)) ** 2)
            / 2
        ).softmax(dim=-1)
        # sanity check: exp(log_prob).sum(dim=-1) == 1 (all, BxD), bin_weights.sum(dim=-1) == 1 (all, BxD)

        # cross entropy loss on each feature
        loss = -(log_prob * bin_weights).sum(
            dim=-1
        )  # loss on each feature, each examples in the batch
        return loss.sum(dim=-1)  # sum along features

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _encode(self, xf, yf):
        yf = 2 * yf - 1
        src = torch.cat((yf, xf), dim=1)
        src_emb = self.positional_encoding(self.real_embed(src))
        src_emb = src_emb.transpose(0, 1)  # Transpose for PyTorch 1.7
        return self.transformer.encoder(src_emb)
        # return self.transformer.encoder(self.positional_encoding(self.real_embed(src)))

    def _decode(self, hist, tgt_mask, y, memory):
        # y = 2*y - 1
        # tgt = torch.cat((y, hist), dim=1)
        # # just returns the logit values of bins in the next feature
        # return self.fc(self.transformer.decoder(self.positional_encoding(self.real_embed(tgt)), memory, tgt_mask))[:,-1]
        y = 2 * y - 1
        tgt = torch.cat((y, hist), dim=1)
        tgt_emb = self.positional_encoding(self.real_embed(tgt))
        tgt_emb = tgt_emb.transpose(0, 1)  # Transpose for PyTorch 1.7
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask)
        output = output.transpose(0, 1)  # Transpose back
        return self.fc(output)[:, -1]

    @torch.no_grad()
    def _sample(self, xf, yf, y, sigma=0.0, temp=1.0):
        self.eval()
        yf = yf.view(-1, 1)
        y = y.view(-1, 1)
        memory = self._encode(xf, yf)  # do this once
        batch_size = xf.shape[0]
        hist = torch.zeros(
            (batch_size, 0), device=xf.device
        )  # initialize empty features, will be concatenated with y

        for i in range(self.num_inputs):
            mask = self.generate_square_subsequent_mask(i + 1)
            logits = temp * self._decode(
                hist, mask, y, memory
            )  # gives unnormalised log probability
            probs = logits.softmax(dim=-1)  # get probability
            sampled_idx = torch.multinomial(probs, 1)  # get bin index
            bin_val = self.bins[sampled_idx]  # get bin value for each batch element
            samp_val = bin_val + sigma * torch.randn_like(
                bin_val
            )  # add sizzle to bin value
            hist = torch.cat((hist, samp_val), dim=1)
        return hist


def train_epoch(model, optimizer, pair_loader, epoch, DEVICE, show_bar=True):
    model.train()
    losses = []
    with tqdm(pair_loader, unit="batch", leave=show_bar) as tepoch:
        for i, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch:04d}")
            xf = batch["x"].to(DEVICE)
            yf = batch["y"].view(-1, 1).to(DEVICE)
            xcf = batch["pair_x"].to(DEVICE)
            ycf = batch["pair_y"].view(-1, 1).to(DEVICE)
            loss = model(xf, yf, xcf, ycf).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tepoch.set_postfix(loss=f"{sum(losses) / len(losses):0.4f}")
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_epoch(model, pair_loader, epoch, DEVICE, show_bar=False):
    model.eval()
    losses = []
    with tqdm(pair_loader, unit="batch", leave=show_bar) as tepoch:
        for i, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch:04d}")
            xf = batch["x"].to(DEVICE)
            yf = batch["y"].view(-1, 1).to(DEVICE)
            xcf = batch["pair_x"].to(DEVICE)
            ycf = batch["pair_y"].view(-1, 1).to(DEVICE)
            loss = model(xf, yf, xcf, ycf).mean()

            losses.append(loss.item())
            tepoch.set_postfix(loss=f"{sum(losses) / len(losses):0.4f}")
    return sum(losses) / len(losses)
