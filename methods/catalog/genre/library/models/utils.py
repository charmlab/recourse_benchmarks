import torch
from torch import nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Batch First, unnecassarily convoluted but works as expected
        x = x + self.pe[: x.size(1), :].view(1, x.size(1), -1)
        return self.dropout(x)
    
class LearnedPositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        # Initialize the embeddings
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        # Batch First, unnecassarily convoluted but works as expected
        x = x + self.pe[: x.size(1), :].view(1, x.size(1), -1)
        return self.dropout(x)

class Embedding2Real(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
    
class Embedding2RealNet(nn.Module):
    """takes in (B,S,D) tensor and returns (B,S) tensor"""
    def __init__(self, input_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # independent encoder for each position
        self.models = nn.ModuleList(
            [Embedding2Real(self.embedding_dim) for _ in range(input_dim)]
        )

    def forward(self, x):
        """Projects each embedding in x on to a real number
        x is of the shape batch_size x seq_len x embed_dim
        Returns a tensor of shape batch_size x seq_len
        """
        # 'dynamic layer length' to support autoregressive generation
        seq_len = x.shape[1]
        assert seq_len <= self.input_dim
        reals = [model(x[:,i]) for i, model in enumerate(self.models[:seq_len])]
        real_out = torch.cat(reals, dim=-1)
        # print(real_out.shape)
        return real_out

class RealEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, src_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(src_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.model(x)
    
class RealEmbeddingNet(nn.Module):

    """takes in (B,S) tensor and returns (B,S,D) tensor"""
    def __init__(self, input_dim: int, embedding_dim: int = 32, src_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.src_dim = src_dim
        # independent encoder for each position
        self.models = nn.ModuleList(
            [RealEmbedding(self.embedding_dim,src_dim=src_dim) for _ in range(input_dim)]
        )

    def forward(self, x):
        """Embeds each feature of x
        x is of the shape batch_size x seq_len
        Returns a tensor of shape batch_size x seq_len x embedding_dim
        """
        # 'dynamic layer length' to support autoregressive generation
        seq_len = x.shape[1]
        assert seq_len <= self.input_dim

        embs = [model(x[:,i].view(-1,self.src_dim)).view(-1,1, self.embedding_dim) for i, model in enumerate(self.models[:seq_len])]
        # embs[0] contains the embedding for y
        embs = torch.cat(embs, dim=1)
        return embs

class HetLinearLayer(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, num_lin:int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_lin = num_lin
        self.layers = nn.ModuleList(
            [nn.Linear(self.input_dim,self.out_dim) for _ in range(num_lin)]
        )

    def forward(self, x):
        """Projects each embedding in x on to a real number
        x is of the shape batch_size x seq_len x embed_dim
        Returns a tensor of shape batch_size x seq_len
        """
        # 'dynamic layer length' to support autoregressive generation
        seq_len = x.shape[1]
        assert seq_len <= self.num_lin
        outs = [layer(x[:,i]).unsqueeze(1) for i, layer in enumerate(self.layers[:seq_len])]
        out = torch.cat(outs, dim=1)
        # print(real_out.shape)
        return out


class ConditionalAttentionForward(nn.Module):
    def __init__(self, dim_x, dim_y, dim_embed,dim_ff, nhead, nlayers):
        super(ConditionalAttentionForward, self).__init__()
        # layer to get emebedding in each dimension
        self.x_real_emb = RealEmbeddingNet(dim_x, dim_embed)
        self.y_real_emb = RealEmbeddingNet(dim_y, dim_embed)
        self.x_emb2real = Embedding2RealNet(dim_x,dim_embed)

        # default vals
        activation = 'relu'
        layer_norm_eps = 1e-5
        norm_first = False
        bias = True

        # Transformer decoder init
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_embed, nhead=nhead,dim_feedforward=dim_ff,activation=activation,
                                                   layer_norm_eps=layer_norm_eps,batch_first=True,norm_first=norm_first, bias=bias)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers, 
                                             norm=nn.LayerNorm(dim_embed, eps=layer_norm_eps, bias=bias))
    def forward(self,x,y):
        x_emb = self.x_real_emb(x)
        y_emb = self.x_real_emb(y)
        decoder_out = self.decoder(x_emb,y_emb)
        out = self.x_emb2real(decoder_out)
        return out

class sklearnBayesOptClf():
    # name sklearn is important to fool viz clf function
    # assumes that func accepts an argument eps to use
    def __init__(self, eps, func, denorm=None):
        self.eps = eps # if data density is exactly zero, eps is considered and classifier 'snaps back to prior'
        self.raw_func = np.vectorize(func)
        if denorm == None:
            self.denorm = lambda x:x
        else:
            self.denorm = denorm
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = self.denorm(x)
        x1 = x[:,0]
        x2 = x[:,1]
        # should return probabilities of each class
        return self.raw_func(x1,x2,self.eps)
    
    def predict_proba(self,x):
        forward_pos = self(x)
        forward_neg = 1 - forward_pos

        return np.stack([forward_neg, forward_pos]).T
    
    def predict(self,x):
        return (self.predict_proba(x)[:,1]>0.5)*1.0
    

class sklearnEpsAdjustedClf():
    # name sklearn is important to fool viz clf function
    # this is used when eps adjustment has to be done by the model and prob_func only accepts x not eps.
    def __init__(self, eps,prob_func,density_func,prior, denorm=None):
        self.eps = eps # if data density is exactly zero, eps is considered and classifier 'snaps back to prior'
        self.prob_func = np.vectorize(prob_func)
        if eps==0:
            self.frac_func = np.vectorize(lambda x1,x2:0)
        else:
            self.frac_func = np.vectorize(lambda x1,x2:eps/(eps+density_func(x1,x2)))
            
        self.prior = prior
        if denorm == None:
            self.denorm = lambda x:x
        else:
            self.denorm = denorm
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = self.denorm(x)
        x1 = x[:,0]
        x2 = x[:,1]
        frac = self.frac_func(x1,x2)
        posterior = self.prob_func(x1,x2)
        adjusted_posterior = frac * self.prior + (1 - frac)*posterior
        return adjusted_posterior
    
    def predict_proba(self,x):
        forward_pos = self(x)
        forward_neg = 1 - forward_pos

        return np.stack([forward_neg, forward_pos]).T
    
    def predict(self,x):
        return (self.predict_proba(x)[:,1]>0.5)*1.0