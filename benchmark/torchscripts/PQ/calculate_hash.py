import torch
import numpy as np

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

def cumulative_sse(X, reverse):
    N, D = X.shape
    if reverse:
        for i in range(N // 2):
            X[i], X[N - i - 1] = X[N - i - 1], X[i]
    out = torch.zeros(N)
    cumX = torch.zeros(D)
    cumX2 = torch.zeros(D)
    out[0] = 0
    for d in range(D):
        cumX[d] = X[0, d]
        cumX2[d] = X[0, d]**2
    for n in range(1, N):
        out[n] = 0
        for d in range(D):
            cumX[d] += X[n, d]
            cumX2[d] += X[n, d]**2
            out[n] += cumX2[d] - (cumX[d]*cumX[d] / (n+1))
    return out
def optimal_split_threshold(B, j):
    X = B
    Xsort, _ = X.sort(dim=0)
    sses_head = cumulative_sse(Xsort, False)
    sses_tail = cumulative_sse(Xsort, True)
    losses = sses_head.clone()
    losses[:-1] += sses_tail[1:]
    n_star = torch.argmin(losses)
    if n_star < Xsort.shape[0] - 1:
        return (Xsort[n_star, j] + Xsort[n_star+1, j]) / 2, losses[n_star]
    else:
        return Xsort[n_star, j], losses[n_star]
def heuristic_select_idxs(B):
    # This function should be implemented according to the specific heuristic of choosing indices
    # As a placeholder, I'm randomly selecting 4 indices
    return torch.randint(0, B[0].shape[1], (4,))
def apply_split(v, B, j):
    mask = B[:, j] < v
    Bbelow = B[mask]
    Babove = B[~mask]
    if len(Bbelow) == 0 or len(Babove) == 0:
        return B, None
    return Bbelow, Babove
def train(A):
    print("Start training on data of shape: ", A.shape)
    buckets = [A]
    j_values = []
    v_values = []
    for t in range(1, 5):  # Up to 4 levels
        from datetime import datetime
        print(t, " ", datetime.now().strftime("%H:%M:%S"))
        new_buckets = []
        J = heuristic_select_idxs(buckets)
        l, j, v = float('inf'), None, [None]*len(buckets)
        for j_ in J:
            l_temp = 0
            v_temp = [None]*len(buckets)
            for i in range(len(buckets)):
                if buckets[i] is not None:  # Check if the bucket is None
                    vi, li = optimal_split_threshold(buckets[i], j_)
                    v_temp[i] = vi
                    l_temp += li
            if l_temp < l:
                l, j, v = l_temp, j_, v_temp
        j_values.append(j)
        v_values.append(v)  # Each v is a list of thresholds at this level
        for i in range(len(buckets)):
            if buckets[i] is not None:  # Check if the bucket is None
                Bbelow, Babove = apply_split(v[i], buckets[i], j)
                new_buckets.append(Bbelow)
                new_buckets.append(Babove if Babove is not None else Bbelow)
        buckets = new_buckets
    return j_values, v_values
# usage:
A = torch.randn(100, 100)
j_values, v_values = train(A)
print("Trained split indices: ", j_values)
print("Trained split thresholds: ", v_values)

# TODO save into this container.pt for pq-hash algo to load
# my_values = {
#     'split_indices': torch.tensor([7845, 2, 4003, 5851], dtype=torch.int32),
#     'v1': torch.tensor([[103.7847]], dtype=torch.float),
#     'v2': torch.tensor([[82.9420, -19.8963]], dtype=torch.float),
#     'v3': torch.tensor([[97.2961, 39.0770, -23.8963, -23.8963]], dtype=torch.float),
#     'v4': torch.tensor([[105.9635, -25.5005, -1.9230, -1.9230, -11.8963, -11.8963, -11.8963, -11.8963]], dtype=torch.float)
# }

# container = torch.jit.script(Container(my_values))
# container.save("container.pt")