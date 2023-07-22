import torch
import time
import os
import math


def hadamard_transform_matrix(n: int) -> torch.Tensor:
    H = torch.tensor([[1.]])
    i = 1
    while i < n:
        H_top = torch.cat((H, H), dim=1)
        H_bottom = torch.cat((H, -H), dim=1)
        H = torch.cat((H_top, H_bottom), dim=0)
        i *= 2
    return H


@torch.jit.script
def FastJLT(A: torch.Tensor, B: torch.Tensor, d: int):
    N, D = A.shape
    M = B.shape[1]

    # Pad A and B for FHT
    log2_D = int(torch.ceil(torch.log2(torch.tensor(D))))
    D_pad = int(2 ** log2_D)
    A_pad = torch.zeros((N, D_pad))
    A_pad[:, :D] = A
    B_pad = torch.zeros((D_pad, M))
    B_pad[:D] = B

    # Construct and apply random signs for each dimension
    randsigns = (torch.randint(0, 2, size=(D_pad,)) * 2 - 1).float()
    randsigns *= 1.0 / torch.sqrt(D_pad)
    A_pad *= randsigns
    B_pad *= randsigns.view(-1, 1)

    # Apply Fast Hadamard Transform
    H = hadamard_transform_matrix(D_pad)
    A_pad = A_pad @ H
    B_pad = H @ B_pad

    # Dimensionality reduction
    keep_prob = log2_D * log2_D / D_pad
    P = (torch.rand((D_pad, d)) > keep_prob).float()
    P *= torch.randn(P.shape[0], P.shape[1]) * (d / keep_prob)
    P *= 1.0 / torch.linalg.norm(P, dim=0)

    return (A_pad @ P) @ (P.t() @ B_pad)


def main():
    width = 1000
    A = torch.rand(10000, width)
    B = torch.rand(width, 5000)

    t = time.time()

    aResult = FastJLT(A, B, 500)
    print("approximate: " + str(time.time() - t) + "s")

    print(aResult)

    # exact result
    t = time.time()
    eResult = torch.matmul(A, B)
    print("\nExact: " + str(time.time() - t) + "s")

    print(eResult)

    print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))

    FastJLT_script = FastJLT.save("FastJLT.pt")


if __name__ == '__main__':
    main()
