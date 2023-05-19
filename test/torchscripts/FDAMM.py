import torch
import time


def get_first_element(tensor):
    if tensor.numel() == 1:
        return tensor.item()
    else:
        return tensor[0].item()


@torch.jit.script
def FDAMM(A: torch.Tensor, B: torch.Tensor, l: int):
    # assert A.shape[1] == B.shape[1]
    mx, n = A.shape
    bn, my = B.shape
    # initialize sketch matrices
    BX = torch.zeros((mx, l))
    BY = torch.zeros((my, l))

    for i in range(n):
        # find zero valued column, to be replaced with a better mechanism
        sum_cols = torch.sum(BX, dim=0)
        zero_valued_columns_X = torch.nonzero(sum_cols == 0).squeeze()  # sum each column to find the zero valued ones

        # if a zero valued column exists, insert a column
        if len(zero_valued_columns_X.shape) != 0:
            idx = int(get_first_element(zero_valued_columns_X))
            BX[:, idx] = A[:, i]

        sum_cols = torch.sum(BY, dim=0)
        zero_valued_columns_Y = torch.nonzero(sum_cols == 0).squeeze()
        if len(zero_valued_columns_Y.shape) != 0:
            idx = int(get_first_element(zero_valued_columns_Y))
            BY[:, idx] = B[i, :]

        # if no zero valued column, shrink accrodingly
        if len(zero_valued_columns_X.shape) == 0 and len(zero_valued_columns_Y.shape) == 0:
            QX, RX = torch.linalg.qr(BX)
            QY, RY = torch.linalg.qr(BY)
            U, SV, V = torch.svd(torch.matmul(RX, RY.t()))

            # find the median of singular values
            S_sorted = torch.sort(SV).values
            delta = (S_sorted[len(S_sorted) // 2] if len(S_sorted) % 2 == 1 else
                     (S_sorted[len(S_sorted) // 2 - 1] + S_sorted[len(S_sorted) // 2]) / 2)

            # shrink the singular values with delta
            # (this is based on co-occuring paper from 2017, diffrent from beta-Co-FD)
            SV_shrunk = torch.clamp(SV - delta, min=0)
            SV = torch.diag_embed(SV_shrunk)
            SV_sqrt = torch.sqrt(SV)

            BX = torch.matmul(torch.matmul(QX, U), SV_sqrt)
            BY = torch.matmul(torch.matmul(QY, V), SV_sqrt)

    return torch.matmul(BX, BY.t())


@torch.jit.script
def RAWMM(A: torch.Tensor, B: torch.Tensor, l: int):
    return torch.matmul(A, B)


def main():
    A = torch.rand(10000, 1000)
    B = torch.rand(1000, 5000)
    # A= A.to('cuda')
    # B = B.to('cuda')
    t = time.time()
    Aresult = FDAMM(A, B, 500)
    print("approximate: " + str(time.time() - t) + "s")
    print(Aresult)

    t = time.time()
    Eresult = torch.matmul(A, B)  # exact result
    print("\nExact: " + str(time.time() - t) + "s")
    print(Eresult)
    FDAMM_script = FDAMM.save("FDAMM.pt")
    RAWMM_script = RAWMM.save("RAWMM.pt")


# print("\nerror: " + str(torch.norm(Aresult - Eresult, p='fro').item()))

if __name__ == '__main__':
    main()
