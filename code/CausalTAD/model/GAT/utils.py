import torch
import pdb

def sp_softmax(indices, values, N):
    _, target = indices
    v_max = values.max()
    exp_v = torch.exp(values - v_max)
    exp_sum = torch.zeros(N, 1, device=indices.device)
    # exp_sum[index[i][j]][j] = value[i][j]
    exp_sum.scatter_add_(0, target.unsqueeze(1), exp_v.unsqueeze(-1))
    exp_sum += 1e-10
    softmax_v = exp_v / exp_sum[target].squeeze(-1)
    return softmax_v

def sp_matmul(indices, values, mat):
    source, target = indices
    out = torch.zeros_like(mat, device=indices.device)
    out.scatter_add_(0, target.expand(mat.size(1), -1).t(), values.unsqueeze(-1) * mat[source])
    return out
