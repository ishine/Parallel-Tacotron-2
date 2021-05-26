import torch
from pt2.modules.duration_predictor import DurationPredictor


def test_duration_predictor():
    import pdb
    pdb.set_trace()
    embed = torch.nn.Embedding(256, 32)
    net = DurationPredictor(32)
    token = torch.randint(0, 255, (1, 10))
    token = embed(token)
    y, ft = net(token)
    assert ft.shape == (1, 10, 32)
    assert y.shape == (1, 10, 1)
