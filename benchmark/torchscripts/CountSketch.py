import torch
import time
import os
import math


@torch.jit.script
def CountSketch(A: torch.Tensor, B: torch.Tensor, s: int):
    m1, n = A.shape
    n, m2 = B.shape
    # initialize sketch matrices
    Ca = torch.zeros((m1, s))
    Cb = torch.zeros((s, m2))

    L = torch.randint(s, (n, ))
    G = torch.randint(2, (n, ))

    # modify the random column with random sign
    for i in range(0, n):
        if G[i] == 1:
            Ca[:, L[i]] += A[:, i]
            Cb[L[i], :] += B[i, :]
        else:
            Ca[:, L[i]] -= A[:, i]
            Cb[L[i], :] -= B[i, :]

    return torch.matmul(Ca, Cb)


def main():
    width = 1000
    A = torch.rand(10000, width)
    B = torch.rand(width, 5000)

    t = time.time()

    aResult = CountSketch(A, B, 500)
    print("approximate: " + str(time.time() - t) + "s")

    print(aResult)

    # exact result
    t = time.time()
    eResult = torch.matmul(A, B)
    print("\nExact: " + str(time.time() - t) + "s")

    print(eResult)

    print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))

    CountSketch_script = CountSketch.save("CountSketch.pt")


if __name__ == '__main__':
    main()
