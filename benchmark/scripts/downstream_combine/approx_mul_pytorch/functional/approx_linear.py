import torch
from ..modules.utils import *

def approx_linear_forward(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu, tag):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA^T + b`.
    the matrix multiply xA^T is approximated
    note: weight transposition is done in this function
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """

    return approx_linear_forward_xA_b(input,weight.t(),bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu, tag)

def approx_linear_forward_xA_b(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu, tag):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA + b`.
    Note: A is assumed not transposed
    the matrix multiply xA is approximated
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(in\_features, out\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    return amm(input, weight, bias, minimal_k, tag)


def amm(A,B,bias, minimal_k, algo):
    algorithms = ["mm", "crs", "crsV2", "countSketch", "bcrs", "ews", "cooFD",
                   "bcooFD", "int8", "int8_fp32", "tugOfWar", "weighted-cr",
                     "smp-pca", "blockLRA", "rip", "fastjlt"]
    if algo not in algorithms:
        algo = "mm"
    torch.ops.AMMBench.setTag(algo)
    C = torch.ops.AMMBench.ammSpecifySs(A, B, minimal_k)
    if bias is not None:
        C += bias
    return C
