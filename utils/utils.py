"""
Utilities function file

@author DisCo Authors
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from .config import Config


class DiscoDataset(Dataset):
    """Dataset wrapper for DisCo training inputs.

    Args:
        xi: N x n_xi input feature matrix (instance features).
        yi: N x yi_dim matrix of label-independent features.
        ya: N x ya_dim matrix of auxiliary features paired with labels/attributes.
        y: N x y_dim matrix of label targets (one-hot or probabilistic).
        a: N-length vector of attribute/group ids.
    """
    def __init__(self, xi, yi, ya, y, a):
        self.xi = xi
        self.yi = yi
        self.ya = ya
        self.y = y
        self.a = a

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "a": torch.as_tensor(self.a[idx], dtype=torch.long),
            "xi": torch.as_tensor(self.xi[idx], dtype=torch.float32),
            "yi": torch.as_tensor(self.yi[idx], dtype=torch.float32),
            "ya": torch.as_tensor(self.ya[idx], dtype=torch.float32),
            "y": torch.as_tensor(self.y[idx], dtype=torch.float32),
        }

def save_object(model, fname, config=None):
    checkpoint = {"model": model.state_dict()}
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, fname)

def load_object(fname, map_location="cpu"):
    try:
        return torch.load(fname, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(fname, map_location=map_location)

def scale_feat(x, a=-1.0, b=1.0):
    x = torch.as_tensor(x, dtype=torch.float32)
    max_x = torch.max(x, dim=1, keepdim=True).values
    min_x = torch.min(x, dim=1, keepdim=True).values
    denom = max_x - min_x
    # Avoid division by zero for constant features (where max_x == min_x)
    # For constant features, return the midpoint of the range [a, b]
    x_prime = torch.where(denom == 0, (a + b) / 2, a + (((x - min_x) * (b - a)) / denom))
    return x_prime.to(torch.float32)

def calc_mode(value_list):
    if len(value_list) == 0:
        return -1, 0
    freq_map = {}
    for i in range(len(value_list)):
        v_i = value_list[i]
        cnt = freq_map.get(v_i)
        if cnt != None:
            cnt = cnt + 1
            freq_map.update({v_i : cnt})
        else:
            freq_map.update({v_i : 1})
    most_freq_v = None
    max_cnt = 0
    for v, cnt in freq_map.items():
        if cnt > max_cnt:
            max_cnt = cnt
            most_freq_v = v
    return most_freq_v, max_cnt

def D_KL(px, qx, keep_batch=False):
    '''
    General KL divergence between probability dist p(x) and q(x), i.e., KL(p||q)
    -> q(x) is the approximating communication channel/distribution
    -> p(x) is the target channel/distribution (we wish to compress)
    <br>
    Notes that this function was derived from:  https://arxiv.org/pdf/1404.2000.pdf
    @author DisCo Authors
    '''
    eps = 1e-6
    px_ = torch.clamp(px, eps, 1.0 - eps)
    log_px = torch.log(px_)
    qx_ = torch.clamp(qx, eps, 1.0 - eps)
    log_qx = torch.log(qx_)

    term1 = torch.sum(-(px_ * log_px), dim=1, keepdim=True)
    term2 = torch.sum(px_ * log_qx, dim=1, keepdim=True)
    KL = -(term1 + term2)
    loss = KL
    if not keep_batch:
        loss = torch.mean(KL) #,axis=0)
    return loss

def D_KL_(qx, px, keep_batch=False):
    '''
    General KL divergence between probability dist q(x) and p(x), i.e., KL(q||p)
    @author DisCo Authors
    '''
    eps = 1e-6
    qx_ = torch.clamp(qx, eps, 1.0 - eps)
    px_ = torch.clamp(px, eps, 1.0 - eps)
    log_qx = torch.log(qx_)
    log_px = torch.log(px_)
    term1 = torch.sum(qx_ * log_qx, dim=1, keepdim=True)
    term2 = torch.sum(qx_ * log_px, dim=1, keepdim=True)
    KL = term1 - term2
    loss = KL
    if not keep_batch:
        loss = torch.mean(KL) #,axis=0)
    return loss

def mse(x_true, x_pred, keep_batch=False):
    '''
    Mean Squared Error
    @author DisCo Authors
    '''
    diff = x_pred - x_true
    se = diff * diff # 0.5 # squared error
    # NLL = -( -se )
    if not keep_batch:
        mse = torch.mean(se)
    else:
        mse = torch.sum(se, dim=-1, keepdim=True)
    return mse

def drop_out(input, rate=0.0, seed=69):
    """
        Custom drop-out function -- returns output as well as binary mask
        -> scale the values of the output by 1/(1-rate) which allows us to just
           set rate to 0 at test time with no further changes needed to compute the
           expectation of the activation output
        @author DisCo Authors
    """
    mask = torch.rand((input.shape[0], input.shape[1]), device=input.device) <= (1.0 - rate)
    mask = mask.to(torch.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask
    
def softmax(x, tau=0.0):
    """
        Softmax function with overflow control built in directly. Contains optional
        temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)
    """
    if tau > 0.0:
        x = x / tau
    max_x = torch.max(x, dim=1, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

def calc_catNLL(target, prob, keep_batch=False):
    """
        Calculates the (negative) Categorical log likelihood under a provided set of probabilities
        for a target (one-hot) label encoding.
    """
    eps = 1e-7
    py = torch.clamp(prob, eps, 1.0 - eps)
    Ly = -torch.sum(torch.log(py) * target, dim=1, keepdim=True)
    if keep_batch is False:
        return torch.mean(Ly)
    return Ly

def sample_gaussian(shape, mu=0.0, sig=1.0):
    """
        Samples a multivariate Gaussian assuming a diagonal covariance
    """
    eps = torch.normal(mean=mu, std=sig, size=shape)
    return eps

def calc_gaussian_KL(mu1, sigSqr1, log_var1, mu2, sigSqr2, log_var2):
    """
        Calculates Kullback-Leibler divergence (KL-D) between two multivariate
        Gaussians strictly assuming each has a diagonal covariance (vector variances).
    """
    eps = 1e-7
    term1 = log_var2 - log_var1 #tf.math.log(sig2) - tf.math.log(sig1)
    diff = (mu1 - mu2)
    term2 = (sigSqr1 + (diff ** 2))/(sigSqr2 * 2 + eps)
    kl = term1 + term2 - 0.5
    return torch.sum(kl, dim=1, keepdim=True)

def calc_gaussian_KL_simple(mu, log_sigma_sqr):
    return -0.5 * torch.sum(1 + log_sigma_sqr - (mu * mu) - torch.exp(log_sigma_sqr), dim=1)

def init_weights(init_type, shape, stddev=1.0):
    if init_type == "he_uniform":
        params = torch.empty(shape)
        torch.nn.init.kaiming_uniform_(params, nonlinearity="relu")
    elif init_type == "he_normal":
        params = torch.empty(shape)
        torch.nn.init.kaiming_normal_(params, nonlinearity="relu")
    elif init_type == "classic_glorot":
        N = (shape[0] + shape[1]) * 1.0
        bound = 4.0 * np.sqrt(6.0/N)
        params = torch.empty(shape).uniform_(-bound, bound)
    elif init_type == "glorot_normal":
        params = torch.empty(shape)
        torch.nn.init.xavier_normal_(params)
    elif init_type == "glorot_uniform":
        params = torch.empty(shape)
        torch.nn.init.xavier_uniform_(params)
    elif init_type == "orthogonal":
        params = torch.empty(shape)
        torch.nn.init.orthogonal_(params, gain=stddev)
    elif init_type == "truncated_normal":
        params = torch.empty(shape)
        torch.nn.init.trunc_normal_(params, std=stddev)
    elif init_type == "normal":
        params = torch.normal(mean=0.0, std=stddev, size=shape)
    else: # alex_uniform
        k = 1.0 / (shape[0] * 1.0) # 1/in_features
        bound = np.sqrt(k)
        params = torch.empty(shape).uniform_(-bound, bound)

    return params

def l1_l2_norm_calculation(theta_y,norm_type,mini_batch_size):
    """
        Normalizes based on L1 and L2 operations. 
    """
    norm_value = 0.0
    for var in theta_y:
        w_norm = torch.linalg.norm(var, ord=norm_type)
        # if norm_type == 1:
        #     w_norm = tf.reduce_sum(tf.math.abs(var))
        # else:
        #     w_norm = tf.reduce_sum(var*var)
        norm_value += (w_norm*w_norm)
    norm_value = norm_value * 1/mini_batch_size
    return norm_value

################################################################################
# Functions for computing empirical KL divergence between two data samples
################################################################################
def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True)
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n
    def interpolate_(x_):
        yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate_

def cumulative_kl(x,y,fraction=0.5):
    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    ex = np.min(dx)
    ey = np.min(dy)
    e = np.min([ex,ey])*fraction
    n = len(x)
    P = ecdf(x)
    Q = ecdf(y)
    KL = (1./n)*np.sum(np.log((P(x) - P(x-e))/(Q(x) - Q(x-e))))
    return KL

def gen_data_plot(Xn, Yn, use_tsne=False, fname="Xi", out_dir=""):
    z_top = Xn
    y_ind = torch.argmax(torch.as_tensor(Yn, dtype=torch.float32), dim=1).cpu().numpy()
    import matplotlib #.pyplot as plt
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.cm.jet
    from sklearn.decomposition import IncrementalPCA
    print(" > Projecting latents via iPCA...")
    if use_tsne is True:
        n_comp = 10 #16 #50
        if z_top.shape[1] < n_comp:
            n_comp = z_top.shape[1] - 2 #z_top.shape[1]-2
            n_comp = max(2, n_comp)
        ipca = IncrementalPCA(n_components=n_comp, batch_size=50)
        ipca.fit(z_top)
        z_2D = ipca.transform(z_top)
        print(" > Finishing project via t-SNE...")
        from sklearn.manifold import TSNE
        z_2D = TSNE(n_components=2,perplexity=30).fit_transform(z_2D)
        #z_2D.shape
    else:
        ipca = IncrementalPCA(n_components=2, batch_size=50)
        ipca.fit(z_top)
        z_2D = ipca.transform(z_top)

    print(" > Plotting 2D encodings...")
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2D[:, 0], z_2D[:, 1], c=y_ind, cmap=cmap)
    plt.colorbar()
    plt.grid()
    plt.savefig("{0}{1}.pdf".format(out_dir,fname))
    plt.clf()


def get_config_file(options):
    # Collect arguments from argv
    cfg_fname = None
    use_gpu = False
    gpu_id = -1
    for opt, arg in options:
        if opt in ("--cfg_fname"):
            cfg_fname = arg.strip()
        elif opt in ("--gpu_id"):
            gpu_id = int(arg.strip())
            use_gpu = True
    mid = gpu_id
    if use_gpu:
        print(" > Using GPU ID {0}".format(mid))
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(mid)
        gpu_tag = '/GPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = '/CPU:0'

    return cfg_fname, gpu_tag


def get_params(cfg_fname):
    args = Config(cfg_fname)
    params = {
        "xi_fname": args.getArg("xi_fname"),
        "yi_fname": args.getArg("yi_fname"),
        "ya_fname": args.getArg("ya_fname"),
        "i_fname": args.getArg("i_fname"),
        "a_fname": args.getArg("a_fname"),
        "y_fname": args.getArg("y_fname"),
        "dev_xi_fname": args.getArg("dev_xi_fname"),
        "dev_yi_fname": args.getArg("dev_yi_fname"),
        "dev_ya_fname": args.getArg("dev_ya_fname"),
        "dev_i_fname": args.getArg("dev_i_fname"),
        "dev_a_fname": args.getArg("dev_a_fname"),
        "dev_y_fname": args.getArg("dev_y_fname"),
        "out_dir": args.getArg("out_dir"),
        "dataset": args.getArg("dataset")
    }

    simulation_params = {
        "n_epoch": int(args.getArg("n_epoch")),  # 100 #200 #500 #200 #50
        "batch_size": int(args.getArg("batch_size")),  # 100 #256 #128 #64 #256 #128
        "save_every": int(args.getArg("save_every")),  # 10
        "eval_every": int(args.getArg("eval_every"))
    }

    disco_model_params = {
        "act_fx": args.getArg("act_fx"),
        "weight_init_scheme": args.getArg("weight_init_scheme"),
        "lat_dim": int(args.getArg("lat_dim")),  # 100 #64 #32
        "opt_type": args.getArg("opt_type"),
        "learning_rate": float(args.getArg("learning_rate")),
        # "max_param_norm": float(args.getArg("max_param_norm")),
        "update_radius": float(args.getArg("update_radius")),
        "lat_fusion_type": args.getArg("lat_fusion_type"),  # "concat" "sum"
        "lat_i_dim": int(args.getArg("lat_i_dim")),  # 100
        "lat_a_dim": int(args.getArg("lat_a_dim")),  # 50
        "drop_p": float(args.getArg("drop_p")),
        "gamma_i": float(args.getArg("gamma_i")),
        "gamma_a": float(args.getArg("gamma_a"))
    }

    return params, simulation_params, disco_model_params


def read_data(params):
    data = {}
    data["Xi"] = np.load(params["xi_fname"], allow_pickle=True)
    data["Y"] = np.load(params["y_fname"], allow_pickle=True)
    gen_data_plot(data["Xi"], data["Y"], use_tsne=False)
    data["Yi"] = np.load(params["yi_fname"])
    data["Ya"] = np.load(params["ya_fname"])
    data["I"] = np.load(params["i_fname"])
    data["A"] = np.load(params["a_fname"])
    data["dev_Xi"] = None
    data["dev_Yi"] = None
    data["dev_Ya"] = None
    data["dev_Y"] = None
    data["dev_I"] = None
    data["dev_A"] = None
    data["train_dataset"] = None
    data["dev_dataset"] = None
    if params["dev_y_fname"] is not None:
        data["dev_Xi"] = np.load(params["dev_xi_fname"])
        data["dev_Yi"] = np.load(params["dev_yi_fname"])
        data["dev_Ya"] = np.load(params["dev_ya_fname"])
        data["dev_Y"] = np.load(params["dev_y_fname"])
        data["dev_I"] = np.load(params["dev_i_fname"])
        data["dev_A"] = np.load(params["dev_a_fname"])
        data["dev_dataset"] = DiscoDataset(
            data["dev_Xi"],
            data["dev_Yi"],
            data["dev_Ya"],
            data["dev_Y"],
            data["dev_A"],
        )
    # automatically count num of total items, total annotators, and get design matrix dimensions
    data["n_i"] = np.max(data["I"]) + 1  # 2000 # number items
    data["n_a"] = np.max(data["A"]) + 1  # 50 # number annotators
    data["yi_dim"] = data["Yi"].shape[1]
    data["ya_dim"] = data["Ya"].shape[1]
    data["y_dim"] = data["Y"].shape[1]
    data["n_xi"] = data["Xi"].shape[1]
    data["train_dataset"] = DiscoDataset(
        data["Xi"],
        data["Yi"],
        data["Ya"],
        data["Y"],
        data["A"],
    )

    print("Xi.shape = ", data["Xi"].shape)
    print("Yi.shape = ", data["Yi"].shape)
    print("Y.shape = ", data["Y"].shape)
    print("Ya.shape = ", data["Ya"].shape)
    print("I.shape = ", data["I"].shape)
    print("A.shape = ", data["A"].shape)

    return data
