import torch
import time
import os
import math


def tug_of_war_mat(m: int, n: int) -> torch.Tensor:
    e = 1 / math.sqrt(m)
    M = torch.randint(2, (m, n))
    return e * (2 * M - 1)


@torch.jit.script
def TugOfWar(A: torch.Tensor, B: torch.Tensor, l: int):
    m, n = A.shape
    n, p = B.shape

    delta = 0.2

    i_iters = int(-math.log(delta))
    j_iters = int(2 * (-math.log(delta) + math.log(-math.log(delta))))

    z = torch.empty((i_iters,))
    AS = []
    SB = []

    for i in range(i_iters):
        S = tug_of_war_mat(l, n)
        SB.append(S.matmul(B))
        AS.append(A.matmul(S.T))

        y = torch.empty((j_iters,))

        for j in range(j_iters):
            Q = tug_of_war_mat(16, p)
            X = A.matmul(B.matmul(Q.T))
            X_hat = AS[i].matmul(SB[i].matmul(Q.T))
            y[j] = torch.norm(X - X_hat) ** 2
        z[i] = torch.median(y)

    i_star = torch.argmin(z)
    return torch.matmul(AS[i_star], SB[i_star])


def main():
    width = 1000
    A = torch.rand(10000, width)
    B = torch.rand(width, 5000)

    t = time.time()

    aResult = TugOfWar(A, B, 500)
    print("approximate: " + str(time.time() - t) + "s")

    print(aResult)

    # exact result
    t = time.time()
    eResult = torch.matmul(A, B)
    print("\nExact: " + str(time.time() - t) + "s")

    print(eResult)

    print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))

    TugOfWar_script = TugOfWar.save("TugOfWar.pt")


if __name__ == '__main__':
    main()
