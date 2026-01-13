import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from losses.disco_loss import DiscoLoss
from model.disco import DISCO


def _make_model(drop_p=0.0):
    return DISCO(
        xi_dim=6,
        yi_dim=4,
        ya_dim=3,
        y_dim=5,
        a_dim=7,
        lat_i_dim=4,
        lat_a_dim=4,
        lat_dim=5,
        drop_p=drop_p,
    )


def _make_concat_model(drop_p=0.0):
    return DISCO(
        xi_dim=6,
        yi_dim=4,
        ya_dim=3,
        y_dim=5,
        a_dim=7,
        lat_i_dim=4,
        lat_a_dim=6,
        lat_dim=5,
        drop_p=drop_p,
        lat_fusion_type="concat",
    )


def test_package_imports():
    import importlib

    disco = importlib.import_module("disco")
    assert hasattr(disco, "DISCO")
    assert hasattr(disco, "DiscoConfig")
    assert hasattr(disco, "DiscoLoss")


def test_forward_shapes_and_device():
    model = _make_model()
    xi = torch.randn(8, 6)
    a = torch.randint(0, 7, (8,))
    y_logits, yi_logits, ya_logits = model(xi, a)
    assert y_logits.shape == (8, 5)
    assert yi_logits.shape == (8, 4)
    assert ya_logits.shape == (8, 3)
    assert y_logits.device == xi.device


def test_dropout_train_eval_behavior():
    model = _make_model(drop_p=0.5)
    xi = torch.randn(4, 6)
    a = torch.randint(0, 7, (4,))

    model.eval()
    out_eval_1 = model(xi, a)[0]
    out_eval_2 = model(xi, a)[0]
    assert torch.allclose(out_eval_1, out_eval_2)

    model.train()
    torch.manual_seed(0)
    out_train_1 = model(xi, a)[0]
    torch.manual_seed(1)
    out_train_2 = model(xi, a)[0]
    assert not torch.allclose(out_train_1, out_train_2)


def test_loss_decreases_on_tiny_batch():
    torch.manual_seed(0)
    model = _make_model(drop_p=0.0)
    loss_fn = DiscoLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    xi = torch.randn(16, 6)
    a = torch.randint(0, 7, (16,))
    y = torch.randint(0, 5, (16,))
    yi = torch.softmax(torch.randn(16, 4), dim=-1)
    ya = torch.softmax(torch.randn(16, 3), dim=-1)

    initial_loss, _ = loss_fn(*model(xi, a), y, yi, ya, model=model)
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        y_logits, yi_logits, ya_logits = model(xi, a)
        loss, _ = loss_fn(y_logits, yi_logits, ya_logits, y, yi, ya, model=model)
        loss.backward()
        optimizer.step()

    final_loss, _ = loss_fn(*model(xi, a), y, yi, ya, model=model)
    assert final_loss.item() < initial_loss.item()


def test_loss_supports_distribution_targets():
    model = _make_model(drop_p=0.0)
    loss_fn = DiscoLoss()
    xi = torch.randn(4, 6)
    a = torch.randint(0, 7, (4,))
    y_dist = torch.softmax(torch.randn(4, 5), dim=-1)
    yi = torch.softmax(torch.randn(4, 4), dim=-1)
    ya = torch.softmax(torch.randn(4, 3), dim=-1)

    loss, _ = loss_fn(*model(xi, a), y_dist, yi, ya, model=model)
    assert torch.isfinite(loss)


def test_encoder_components_shapes():
    model = _make_model()
    xi = torch.randn(3, 6)
    a = torch.randint(0, 7, (3,))
    z_i = model.encode_i(xi)
    z_a = model.encode_a(a)
    z = model.encode(xi, a)
    assert z_i.shape == (3, 4)
    assert z_a.shape == (3, 4)
    assert z.shape == (3, 5)


def test_concat_fusion_and_decoders():
    model = _make_concat_model()
    xi = torch.randn(2, 6)
    a = torch.randint(0, 7, (2,))
    z = model.encode(xi, a)
    yi_logits = model.decode_yi(z)
    ya_logits = model.decode_ya(z)
    y_logits = model.decode_y(z)
    assert z.shape == (2, 5)
    assert yi_logits.shape == (2, 4)
    assert ya_logits.shape == (2, 3)
    assert y_logits.shape == (2, 5)


def test_decode_y_ensemble_restores_training_mode():
    model = _make_model(drop_p=0.1)
    model.train()
    xi = torch.randn(1, 6)
    y_prob, y_logits = model.decode_y_ensemble(xi)
    assert model.training is True
    assert y_prob.shape == (7, 5)
    assert y_logits.shape == (7, 5)
    assert torch.allclose(y_prob.sum(dim=-1), torch.ones(7))
