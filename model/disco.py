import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from ..utils import init_weights


@dataclass
class DiscoConfig:
    xi_dim: int
    yi_dim: int
    ya_dim: int
    y_dim: int
    a_dim: int
    lat_i_dim: int = 20
    lat_a_dim: int = 30
    lat_dim: int = 10
    act_fx: str = "softsign"
    init_type: str = "gaussian"
    name: str = "disco"
    i_dim: int = -1
    lat_fusion_type: str = "sum"
    drop_p: float = 0.0
    gamma_i: float = 1.0
    gamma_a: float = 1.0
    l1_norm: float = 0.0
    l2_norm: float = 0.0

    def to_dict(self):
        return {
            "xi_dim": self.xi_dim,
            "yi_dim": self.yi_dim,
            "ya_dim": self.ya_dim,
            "y_dim": self.y_dim,
            "a_dim": self.a_dim,
            "lat_i_dim": self.lat_i_dim,
            "lat_a_dim": self.lat_a_dim,
            "lat_dim": self.lat_dim,
            "act_fx": self.act_fx,
            "init_type": self.init_type,
            "name": self.name,
            "i_dim": self.i_dim,
            "lat_fusion_type": self.lat_fusion_type,
            "drop_p": self.drop_p,
            "gamma_i": self.gamma_i,
            "gamma_a": self.gamma_a,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
        }


class DISCO(nn.Module):
    """
        The proposed DisCo model. See paper for details.

        @author DisCo Authors
    """

    def __init__(
        self,
        xi_dim,
        yi_dim,
        ya_dim,
        y_dim,
        a_dim,
        lat_i_dim=20,
        lat_a_dim=30,
        lat_dim=10,
        act_fx="softsign",
        init_type="gaussian",
        name="disco",
        i_dim=-1,
        lat_fusion_type="sum",
        drop_p=0.0,
        gamma_i=1.0,
        gamma_a=1.0,
        l1_norm=0.0,
        l2_norm=0.0,
    ):
        super().__init__()
        self.name = name
        self.gamma_i = gamma_i
        self.gamma_a = gamma_a
        self.lat_fusion_type = lat_fusion_type
        self.i_dim = i_dim
        self.a_dim = a_dim
        self.y_dim = y_dim
        self.xi_dim = xi_dim
        self.yi_dim = yi_dim
        self.ya_dim = ya_dim
        self.lat_dim = lat_dim
        self.lat_i_dim = lat_i_dim
        self.lat_a_dim = lat_a_dim
        self.drop_p = drop_p
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm

        if self.lat_fusion_type != "concat":
            self.lat_a_dim = self.lat_i_dim
            print(" > Setting lat_a.dim equal to lat_i.dim (dim = {0})".format(self.lat_i_dim))

        self.act_fx = act_fx
        self.fx = None
        if act_fx == "tanh":
            self.fx = torch.tanh
        elif act_fx == "sigmoid":
            self.fx = torch.sigmoid
        elif act_fx == "relu":
            self.fx = F.relu
        elif act_fx == "relu6":
            self.fx = F.relu6
        elif act_fx == "lrelu":
            self.fx = F.leaky_relu
        elif act_fx == "elu":
            self.fx = F.elu
        elif act_fx == "identity":
            self.fx = lambda x: x
        else:
            print(" > Choosing base DisCo activation function - softsign(.)")
            self.fx = F.softsign

        bot_dim = self.lat_i_dim
        if self.lat_fusion_type == "concat":
            bot_dim = self.lat_i_dim + self.lat_a_dim

        self.item_proj = nn.Linear(self.xi_dim, self.lat_i_dim, bias=False)
        self.annotator_emb = nn.Embedding(self.a_dim, self.lat_a_dim)
        self.fusion_proj = nn.Linear(bot_dim, self.lat_dim, bias=False)
        self.encoder_proj = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.y_head = nn.Linear(self.lat_dim, self.y_dim, bias=False)
        self.yi_head = nn.Linear(self.lat_dim, self.yi_dim, bias=False)
        self.ya_head = nn.Linear(self.lat_dim, self.ya_dim, bias=False)
        self.dropout = nn.Dropout(p=self.drop_p)

        self._init_weights(init_type)

    def _init_weights(self, init_type, stddev=0.05):
        seed = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                self.item_proj.weight.copy_(init_weights(init_type, [self.lat_i_dim, self.xi_dim], stddev=stddev))
                self.annotator_emb.weight.copy_(init_weights(init_type, [self.a_dim, self.lat_a_dim], stddev=stddev))
                self.fusion_proj.weight.copy_(
                    init_weights(init_type, [self.lat_dim, self.fusion_proj.in_features], stddev=stddev)
                )
                self.encoder_proj.weight.copy_(
                    init_weights(init_type, [self.lat_dim, self.lat_dim], stddev=stddev)
                )
                self.y_head.weight.copy_(init_weights(init_type, [self.y_dim, self.lat_dim], stddev=stddev))
                self.yi_head.weight.copy_(init_weights(init_type, [self.yi_dim, self.lat_dim], stddev=stddev))
                self.ya_head.weight.copy_(init_weights(init_type, [self.ya_dim, self.lat_dim], stddev=stddev))

    def encode_i(self, xi):
        """
            Calculates projection/embedding of item feature vector x_i
        """
        return self.item_proj(xi)

    def encode_a(self, a):
        """
            Calculates projection/embedding of annotator a
        """
        av = a.long().view(-1)
        z_enc = self.annotator_emb(av)
        if len(z_enc.shape) < 2:
            z_enc = z_enc.unsqueeze(0)
        return z_enc

    def encode(self, xi, a):
        if self.lat_fusion_type == "concat":
            z = self.fx(torch.cat([self.encode_i(xi), self.encode_a(a)], dim=1))
        else:
            z = self.fx(self.encode_i(xi) + self.encode_a(a))
        z = self.transform(z)
        return z

    def transform(self, z):
        z_p = self.fx(self.fusion_proj(z))
        z_p = self.dropout(z_p)
        z_e = self.fx(self.encoder_proj(z_p) + z_p)
        z_e = self.dropout(z_e)
        return z_e

    def decode_yi(self, z):
        return self.yi_head(z)

    def decode_ya(self, z):
        return self.ya_head(z)

    def decode_y(self, z):
        return self.y_head(z)

    def forward(self, xi, a):
        z = self.encode(xi, a)
        y_logits = self.decode_y(z)
        yi_logits = self.decode_yi(z)
        ya_logits = self.decode_ya(z)
        return y_logits, yi_logits, ya_logits

    def decode_y_ensemble(self, xi):
        """
            Computes the label distribution given only an item feature vector
            (and model's knowledge of all known annotators).
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            z_i = self.encode_i(xi)
            annotator_ids = torch.arange(self.a_dim, device=xi.device)
            z_a = self.annotator_emb(annotator_ids)
            if self.lat_fusion_type == "concat":
                tiled_z_i = z_i.expand(z_a.shape[0], -1)
                z = self.fx(torch.cat([tiled_z_i, z_a], dim=1))
            else:
                z = self.fx(z_a + z_i)
            z = self.transform(z)
            y_logits = self.decode_y(z)
            y_prob = torch.softmax(y_logits, dim=-1)
        finally:
            if was_training:
                self.train()
        return y_prob, y_logits

    def infer_a(self, xi, yi, K, beta, gamma=0.0, is_verbose=False):
        """
            Infer an annotator embedding given only an item feature and label
            distribution vector pair.
        """
        print("WARNING: DO NOT USE THIS! NOT DEBUGGED FOR CONCAT AT THE MOMENT!")
        best_L = None
        batch_size = yi.shape[0]
        z_eps = 0.0
        if "elu" in self.act_fx:
            z_eps = 0.001
        z_i = self.encode_i(xi)
        z_a = torch.zeros([batch_size, self.lat_dim], device=xi.device) + z_eps
        for k in range(K):
            z_a.requires_grad_()
            z = self.fx(z_i + z_a)
            z = self.transform(z)
            yi_logits = self.decode_yi(z)
            yi_log_prob = F.log_softmax(yi_logits, dim=-1)
            Lyi = F.kl_div(yi_log_prob, yi, reduction="batchmean") * self.gamma_i
            if is_verbose is True:
                print("k({0}) KL(p(yi)||yi) = {1}".format(k, Lyi))
            if best_L is not None:
                if Lyi < best_L:
                    best_L = Lyi
                else:
                    break
            else:
                best_L = Lyi
            d_z_a = torch.autograd.grad(Lyi, z_a)[0]
            z_a = (z_a - d_z_a * beta - z_a * gamma).detach()
        return z_a

    def clear(self):
        return
