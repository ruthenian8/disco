import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path.insert(0, '../utils/')
sys.path.append('../utils')
from utils.utils import softmax, init_weights, calc_catNLL, D_KL, drop_out, l1_l2_norm_calculation

class DISCO(nn.Module):
    """
        The proposed DisCo model. See paper for details.

        @author DisCo Authors
    """
    def __init__(self, xi_dim, yi_dim, ya_dim, y_dim, a_dim, lat_i_dim=20, lat_a_dim=30,
                 lat_dim=10, act_fx="softsign", init_type="gaussian", name="disco",
                 i_dim=-1, lat_fusion_type="sum", drop_p=0.0, gamma_i=1.0, gamma_a=1.0,
                 l1_norm=0.0, l2_norm=0.0, device=None):
        super().__init__()
        self.name = name
        self.seed = 69
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma_i = gamma_i #1.0
        self.gamma_a = gamma_a #1.0
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
        self.drop_p = drop_p #0.5
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
            self.fx = F.softsign # hidden layer activation function
        self.fx_y = softmax
        self.fx_yi = softmax
        self.fx_ya = softmax

        stddev = 0.05 # 0.025
        self.theta_y = nn.ParameterList()

        self.Wi = nn.Parameter(init_weights(init_type, [self.xi_dim, self.lat_i_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wi)

        self.Wa = nn.Parameter(init_weights(init_type, [self.a_dim, self.lat_a_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wa)

        bot_dim = self.lat_i_dim
        if self.lat_fusion_type == "concat":
            bot_dim = self.lat_i_dim + self.lat_a_dim

        self.Wp = nn.Parameter(init_weights(init_type, [bot_dim, self.lat_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wp)

        self.We = None
        #if collapse_We is False:
        self.We = nn.Parameter(init_weights(init_type, [self.lat_dim, self.lat_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.We)

        self.Wy = nn.Parameter(init_weights(init_type, [self.lat_dim, self.y_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wy)

        self.Wyi = nn.Parameter(init_weights(init_type, [self.lat_dim, self.yi_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wyi)

        self.Wya = nn.Parameter(init_weights(init_type, [self.lat_dim, self.ya_dim], self.seed, stddev=stddev))
        self.theta_y.append(self.Wya)

        self.register_buffer("z_i", torch.zeros([1, self.lat_dim]))
        self.register_buffer("z_a", torch.zeros([1, self.lat_dim]))

        self.eta_v = 0.002
        self.moment_v = 0.9
        adam_eps = 1e-7 #1e-8  1e-6
        self.y_opt = None
        self.y_opt = torch.optim.Adam(self.theta_y, lr=self.eta_v, betas=(0.9, 0.999), eps=adam_eps)
        self.to(self.device)

    def set_opt(self, opt_type, eta_v, moment_v=0.9):
        adam_eps = 1e-7
        self.eta_v = eta_v
        self.moment_v = moment_v
        if opt_type == "adam":
            self.y_opt = torch.optim.Adam(self.theta_y, lr=self.eta_v, betas=(0.9, 0.999), eps=adam_eps)
        elif opt_type == "rmsprop":
            self.y_opt = torch.optim.RMSprop(self.theta_y, lr=self.eta_v, alpha=0.9, momentum=self.moment_v, eps=1e-6)
        else:
            self.y_opt = torch.optim.SGD(self.theta_y, lr=self.eta_v)

    def calc_loss(self, y, yi, ya, y_prob, yi_prob, ya_prob):
        Ly = calc_catNLL(target=y,prob=y_prob,keep_batch=True) #Ly = D_KL(y_prob, y)
        Ly = torch.mean(Ly) #Ly = torch.sum(Ly)
        Lyi = D_KL(yi, yi_prob) * self.gamma_i
        Lya = D_KL(ya, ya_prob) * self.gamma_a
        l1 = 0.0
        l2 = 0.0
        #NS = # of rows in the mini batch (rows in Y)
        mini_bath_size = y.shape[0] * 1.0
        if self.l1_norm > 0:
            l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm 
        if self.l2_norm > 0:
            l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm 
        L_t = Ly + Lyi + Lya + l1 + l2
        return L_t, Ly, Lyi, Lya

    def encode_i(self, xi):
        """
            Calculates projection/embedding of item feature vector x_i
        """
        z_enc = torch.matmul(xi, self.Wi)
        return z_enc

    def encode_a(self, a):
        """
            Calculates projection/embedding of annotator a
        """
        av = torch.as_tensor(a, dtype=torch.long, device=self.device)
        av = av.squeeze()
        # Ensure annotator indices are within the valid range [0, a_dim)
        if av.numel() > 0:
            a_dim = self.Wa.size(0)
            if torch.any(av < 0) or torch.any(av >= a_dim):
                min_idx = torch.min(av)
                max_idx = torch.max(av)
                raise IndexError(
                    f"Annotator index out of range: observed min={min_idx.item()}, "
                    f"max={max_idx.item()}, but valid range is [0, {a_dim - 1}]."
                )
        z_enc = self.Wa[av]
        if len(z_enc.shape) < 2:
            z_enc = z_enc.unsqueeze(0)
        return z_enc

    def encode(self, xi, a):
        z = None
        if self.lat_fusion_type == "concat":
            z = self.fx(torch.cat([self.encode_i(xi), self.encode_a(a)], dim=1))
        else:
            z = self.fx(self.encode_i(xi) + self.encode_a(a))
        z = self.transform(z)
        return z

    def transform(self,z):
        z_p = self.fx(torch.matmul(z, self.Wp))
        if self.drop_p > 0.0:
            z_p, _ = drop_out(z_p, rate=self.drop_p)
        z_e = self.fx(torch.matmul(z_p, self.We) + z_p)
        if self.drop_p > 0.0:
            z_e, _ = drop_out(z_e, rate=self.drop_p)
        return z_e

    def decode_yi(self, z):
        y_logits = torch.matmul(z, self.Wyi)
        y_dec = self.fx_yi(y_logits)
        return y_dec, y_logits

    def decode_ya(self, z):
        y_logits = torch.matmul(z, self.Wya)
        y_dec = self.fx_ya(y_logits)
        return y_dec, y_logits

    def decode_y(self, z):
        y_logits = torch.matmul(z, self.Wy)
        y_dec = self.fx_y(y_logits)
        return y_dec, y_logits

    def update(self, xi, a, yi, ya, y, update_radius=-1.):
        """
            Updates model parameters given data batch (i, a, yi, ya, y)
        """
        self.y_opt.zero_grad()
        z = self.encode(xi, a)
        yi_prob, _ = self.decode_yi(z)
        ya_prob, _ = self.decode_ya(z)
        y_prob, _ = self.decode_y(z)

        Ly = calc_catNLL(target=y, prob=y_prob, keep_batch=True)
        Ly = torch.mean(Ly)

        Lyi = D_KL(yi, yi_prob) * self.gamma_i
        Lya = D_KL(ya, ya_prob) * self.gamma_a

        l1 = 0.0
        l2 = 0.0
        mini_batch_size = y.shape[0] * 1.0
        if self.l1_norm > 0:
            l1 = l1_l2_norm_calculation(self.theta_y, 1, mini_batch_size) * self.l1_norm
        if self.l2_norm > 0:
            l2 = l1_l2_norm_calculation(self.theta_y, 2, mini_batch_size) * self.l2_norm
        L_t = Ly + Lyi + Lya + l1 + l2

        L_t.backward()
        # apply optional gradient clipping
        if update_radius > 0.0:
            for param in self.theta_y:
                if param.grad is not None:
                    param.grad.data.clamp_(-update_radius, update_radius)
        # update parameters given derivatives
        self.y_opt.step()
        return L_t.detach()

    def decode_y_ensemble(self, xi):
        """
            Computes the label distribution given only an item feature vector
            (and model's knowledge of all known annotators).
        """
        drop_p = self.drop_p + 0
        self.drop_p = 0.0 # turn off dropout

        z_i = self.encode_i(xi)
        z_a = self.Wa + 0 # gather all known annotators
        z = None
        if self.lat_fusion_type == "concat":
            tiled_z_i = z_i.expand(z_a.shape[0], -1) # smear z_i across row dim of z_a (ensure same shapes)
            z = self.fx(torch.cat([tiled_z_i, z_a], dim=1))
        else:
            z = self.fx(z_a + z_i)
        z = self.transform(z)
        y_prob, y_logits = self.decode_y(z)

        self.drop_p = drop_p # turn dropout back on
        return y_prob, y_logits

    def infer_a(self, xi, yi, K, beta, gamma=0.0, is_verbose=False):
        """
            Infer an annotator embedding given only an item feature and label
            distribution vector pair.
        """
        print("WARNING: DO NOT USE THIS! NOT DEBUGGED FOR CONCAT AT THE MOMENT!")
        best_L = None
        batch_size = yi.shape[0]
        z_eps = 0.0 #0.001
        if "elu" in self.act_fx:
            z_eps = 0.001
        # Step 1: encode xi
        z_i = self.encode_i(xi)
        self.z_a = torch.zeros([batch_size, self.lat_dim], device=self.device) + z_eps
        # Step 2: find za given xi, yi
        for k in range(K):
            self.z_a.requires_grad_()
            z = self.fx(z_i + self.z_a)
            z = self.transform(z)
            yi_prob, _ = self.decode_yi(z)
            Lyi = D_KL(yi, yi_prob) * self.gamma_i
            Lyi = torch.sum(Lyi)
            if is_verbose is True:
                print("k({0}) KL(p(yi)||yi) = {1}".format(k, Lyi))
            # check early halting criterion
            if best_L is not None:
                if Lyi < best_L:
                    best_L = Lyi
                else:
                    break # early stop at this point
            else:
                best_L = Lyi
            d_z_a = torch.autograd.grad(Lyi, self.z_a)[0] # get KL gradient w.r.t. z_a
            self.z_a = (self.z_a - d_z_a * beta - self.z_a * gamma).detach() # update latent z_a
        z_a = self.z_a
        return z_a

    def clear(self):
        self.z_i.zero_()
        self.z_a.zero_()

    def to(self, device):
        # First, delegate device transfer of parameters/buffers to nn.Module
        module = super().to(device)
        # Then, update internal device reference
        self.device = torch.device(device)
        # Finally, move optimizer state tensors to the new device
        if self.y_opt is not None:
            for state in self.y_opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        return module
