import torch


def compute_covariance(x):
    _xtx = x.transpose(0, 1)
    _xtx = _xtx - _xtx.mean(1, True)
    xtx = (torch.matmul(_xtx, _xtx.transpose(0, 1))).abs() / _xtx.shape[1]
    # xtx = torch.sqrt((torch.matmul(out2d_u, out2d_u.transpose(0, 1))).abs() / out2d_u.shape[1])
    return xtx
