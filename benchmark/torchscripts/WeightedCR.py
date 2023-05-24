import torch
import time

@torch.jit.script
def CR(A: torch.Tensor, B: torch.Tensor, c: int) -> torch.Tensor:
    """CR algorithm https://www.stat.berkeley.edu/~mmahoney/pubs/matrix1_SICOMP.pdf

    Args:
        A (torch.Tensor): matrix A
        B (torch.Tensor): matrix B
        c (int): number of sampling

    Returns:
        torch.Tensor: CR approximation matrix
    """
    torch.manual_seed(0)

    _, n = A.shape
    
    # probability distribution
    probability_distribution = torch.zeros((n))
    for i in range(n):
        probability_distribution[i] = torch.norm(A.T[i], p='fro')*torch.norm(B[i], p='fro')
    probability_distribution /= probability_distribution.sum()
    
    # S
    S = torch.zeros((n, c))
    sample_indices = torch.multinomial(probability_distribution, c, replacement=True)
    
    for trial, index in enumerate(sample_indices):
        S[int(index.item())][trial]=1
        
    # D
    D = torch.diag(1/torch.sqrt(c*probability_distribution[sample_indices]))

    # ASD(SD)^TB
    SS = torch.matmul(S, D)
    C = torch.matmul(A, SS)
    R = torch.matmul(SS.T, B)
    CR = torch.matmul(C, R)
    
    return CR


@torch.jit.script
def weighted_CR(A: torch.Tensor, B: torch.Tensor, c: int):
    """weighted CR algorithm https://arxiv.org/abs/2011.09709

    Args:
        A (torch.Tensor): matrix A
        B (torch.Tensor): matrix B
        c (int): number of sampling

    Returns:
        torch.Tensor: weighted CR approximation matrix
    """
    torch.manual_seed(0)

    _, n = A.shape
    
    # probability distribution
    probability_distribution = torch.zeros((n))
    for i in range(n):
        probability_distribution[i] = torch.norm(A.T[i], p='fro')*torch.norm(B[i], p='fro')
    probability_distribution /= probability_distribution.sum()

    # S
    sample_indices = torch.multinomial(probability_distribution, c, replacement=True)
    unique_indices, occurences = torch.unique(sample_indices, return_counts=True)

    S = torch.zeros((n, len(unique_indices)))

    for trial, index in enumerate(unique_indices):
        S[int(index.item())][trial]=1

    # D
    D = torch.diag(torch.sqrt(occurences)/torch.sqrt(c*probability_distribution[unique_indices]))

    # ASD(SD)^TB
    SS = torch.matmul(S, D)
    C = torch.matmul(A, SS)
    R = torch.matmul(SS.T, B)
    weighted_CR = torch.matmul(C, R)
    
    return weighted_CR


def main():
    A = torch.rand(10000, 1000)
    B = torch.rand(1000, 5000)
    c = 100
    
    t = time.time()
    AB = torch.matmul(A, B)
    print("AB time: ", time.time() - t)
    print("AB fro: ", torch.norm(AB, p='fro'))
    
    t = time.time()
    
    CR_result = CR(A, B, c)
    print("CR time: ", time.time() - t)
    print("CR error: ", torch.norm(AB-CR_result, p='fro'))

    t = time.time()
    weighted_CR_result = weighted_CR(A, B, c)
    print("weighted_CR time: ", time.time() - t)
    print("weighted_CR error: ", torch.norm(AB-weighted_CR_result, p='fro'))
    
    # CR.save("CR.pt")
    weighted_CR.save("weighted_CR.pt")

if __name__ == '__main__':
    main()
