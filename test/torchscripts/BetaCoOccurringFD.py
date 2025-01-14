import torch
import time
import os
import math


def get_first_element(tensor):
    if tensor.numel() == 1:
        return tensor.item()
    else:
        return tensor[0].item()


def is_empty_tensor(tensor):
    return tensor.numel() == 0


def attenuate(beta: float, k: torch.Tensor, l: int) -> torch.Tensor:
    return (torch.exp(k * beta / (l - 1)) - 1) / (torch.exp(beta) - 1)


@torch.jit.script
def FDAMM(A: torch.Tensor, B: torch.Tensor, l: int):
    B = B.t()
    beta = 1.0
    assert A.shape[1] == B.shape[1]
    mx, n = A.shape
    my, n = B.shape
    # initialize sketch matrices
    BX = torch.zeros((mx, l))
    BY = torch.zeros((my, l))

    # the first l iterations
    for i in range(l):
        BX[:, i] = A[:, i]
        BY[:, i] = B[:, i]

    zero_columns = torch.tensor([0])
    zero_columns = zero_columns[1:]

    # iteration l to n: insert if available, else shrink sketch matrices
    for i in range(l, n):
        # acruire the index of a zero valued column
        if len(zero_columns) != 0:
            idx = int(get_first_element(zero_columns))
            # assert idx == get_first_element(torch.nonzero(torch.sum(BX, dim = 0) == 0).squeeze())
            BX[:, idx] = A[:, i]
            BY[:, idx] = B[:, i]
            zero_columns = zero_columns[1:]

        # if no zero valued column, shrink accrodingly
        else:
            QX, RX = torch.linalg.qr(BX)
            QY, RY = torch.linalg.qr(BY)
            U, SV, V = torch.svd(torch.matmul(RX, RY.t()))

            # find the median of singular values
            S_sorted = torch.sort(SV).values
            delta = (S_sorted[len(S_sorted) // 2] if len(S_sorted) % 2 == 1 else
                     (S_sorted[len(S_sorted) // 2 - 1] + S_sorted[len(S_sorted) // 2]) / 2)

            # delta = S_sorted[-1]

            # parameterizedReduceRank
            indices = torch.arange(l, dtype=torch.float32)
            attenuated_values = attenuate(beta, indices, l)
            parameterizedReduceRank = delta * attenuated_values
            SV_shrunk = torch.clamp(SV - parameterizedReduceRank, min=0)

            # restore SV diagnal matrix
            SV = torch.diag_embed(SV_shrunk)
            SV_sqrt = torch.sqrt(SV)

            # update indices of zero valued columns
            zero_indices = torch.nonzero(SV_shrunk == 0).squeeze()
            zero_columns = torch.unique(torch.cat((zero_columns, zero_indices)))

            # update sketch matrices
            BX = torch.matmul(torch.matmul(QX, U), SV_sqrt)
            BY = torch.matmul(torch.matmul(QY, V), SV_sqrt)

    return torch.matmul(BX, BY.t())


def main():
    width = 1000
    A = torch.rand(10000, width)
    B = torch.rand(width, 5000)

    t = time.time()

    aResult = FDAMM(A, B, 500)
    print("approximate: " + str(time.time() - t) + "s")

    print(aResult)

    # exact result
    t = time.time()
    eResult = torch.matmul(A, B)
    print("\nExact: " + str(time.time() - t) + "s")

    print(eResult)

    print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))

    FDAMM_script = FDAMM.save("BetaCoOccurringFD.pt")


if __name__ == '__main__':
    main()
