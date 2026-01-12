import torch
import torch.nn as nn
import torch.nn.functional as F


def regularization_loss(model, l1_norm=0.0, l2_norm=0.0, batch_size=1.0):
    l1 = 0.0
    l2 = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Do not regularize bias terms. Biases are usually 1D, and are also
        # covered by the ndim check below; this explicit name check is kept
        # for clarity and in case non-1D biases are ever introduced.
        if name.endswith("bias"):
            continue
        # Do not regularize annotator embeddings. These parameters encode
        # per-annotator identifiers rather than model weights; applying L1/L2
        # regularization to them has been found to hurt performance.
        if "annotator_emb" in name:
            continue
        # Skip scalar and 1D parameters (e.g., most biases and scaling factors)
        # from L1/L2 regularization; we only regularize higher-dimensional
        # weight tensors.
        if param.ndim <= 1:
            continue
        if l1_norm > 0:
            l1 = l1 + param.abs().sum()
        if l2_norm > 0:
            l2 = l2 + param.pow(2).sum()
    if l1_norm > 0:
        l1 = l1_norm * (l1 / batch_size)
    if l2_norm > 0:
        l2 = l2_norm * (l2 / batch_size)
    return l1, l2


class DiscoLoss(nn.Module):
    def __init__(self, gamma_i=1.0, gamma_a=1.0, l1_norm=0.0, l2_norm=0.0):
        super().__init__()
        self.gamma_i = gamma_i
        self.gamma_a = gamma_a
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_logits, yi_logits, ya_logits, y, yi, ya, model=None):
        y_log_prob = F.log_softmax(y_logits, dim=-1)
        if y.dtype == torch.long and y.dim() == 1:
            Ly = self.ce_loss(y_logits, y)
        else:
            Ly = self.kl_div(y_log_prob, y)

        yi_log_prob = F.log_softmax(yi_logits, dim=-1)
        ya_log_prob = F.log_softmax(ya_logits, dim=-1)
        Lyi = self.kl_div(yi_log_prob, yi) * self.gamma_i
        Lya = self.kl_div(ya_log_prob, ya) * self.gamma_a

        l1 = 0.0
        l2 = 0.0
        if model is not None and (self.l1_norm > 0 or self.l2_norm > 0):
            l1, l2 = regularization_loss(model, self.l1_norm, self.l2_norm, batch_size=y.shape[0])
        loss = Ly + Lyi + Lya + l1 + l2
        metrics = {
            "Ly": Ly.detach(),
            "Lyi": Lyi.detach(),
            "Lya": Lya.detach(),
            "l1": torch.as_tensor(l1).detach(),
            "l2": torch.as_tensor(l2).detach(),
        }
        return loss, metrics
