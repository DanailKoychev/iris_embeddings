import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable


class Stretch(nn.Module):
    def __init__(self, d):
        super(Stretch, self).__init__()
        self.pows = nn.Parameter(torch.ones(d))
        self.ws = nn.Parameter(torch.ones(d))
        self.bs = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        sign = (x >= 0) * 2 - 1
        return (torch.abs(x) ** self.pows) * sign * self.ws + self.bs


def get_pip_distance_from_array(x, y):
    x = x / torch.norm(x)
    y = y / torch.norm(y)

    x_tr = torch.transpose(x, 1, 0)
    y_tr = torch.transpose(y, 1, 0)

    pip_x = torch.matmul(x, x_tr)
    pip_y = torch.matmul(y, y_tr)

    pip_loss = torch.norm((pip_x - pip_y))
    return pip_loss


def optimize_global_geometry(matrix, ref_matrix, iters=500, lr=5e-2, device='cuda'):
    '''Finds the best transform (x^p)*w + b to apply to ever row of "matrix" to fit
    the high-level geometry of "ref_matrix"'''
    matrix = torch.tensor(matrix).float().to(device)
    ref_matrix = torch.tensor(ref_matrix).float().to(device)
    transformation = Stretch(matrix.shape[-1]).to(device)
    opt = torch.optim.Adam(transformation.parameters(), lr=lr)

    losses = []
    for _ in range(iters):
        transformed = (transformation(matrix))
        loss = get_pip_distance_from_array(transformed, ref_matrix)
        losses.append(float(loss.detach().cpu()))

        opt.zero_grad()
        loss.backward()
        opt.step()

    p, w, b = np.array(([
        param.detach().cpu().numpy()
        for param in transformation.parameters()]))
    return {'p': p, 'w': w, 'b': b}, transformation, losses

