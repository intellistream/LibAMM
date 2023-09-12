import torch
from sklearn.cluster import KMeans
from torch import nn

def getCodewordAndLookUpTable(A, B, m):
    
    # Sample matrix sizes and PQ parameters
    A_rows, A_cols = A.shape
    lA = A_rows // m // 10  # Number of centroids for each subspace
    CA = A_cols // m  # Dimension of each subspace

    # Sample matrix sizes and PQ parameters
    B_rows, B_cols = B.shape
    lB = B_cols // m // 10  # Number of centroids for each subspace
    CB = B_rows // m  # Dimension of each subspace

    # Initialize lists to store subspaces and centroids
    subspaces_A = []
    codewords_A = []

    # Create subspaces and centroids
    for i in range(m):
        subspace_A = A[:, i * CA : (i + 1) * CA]  # 500*20
        subspaces_A.append(subspace_A)

        # Apply KMeans on the row vectors within the subspace
        kmeans = KMeans(n_clusters=lA, n_init=10)
        kmeans.fit(subspace_A)  # 500*20
        subspace_centroids_A = torch.tensor(kmeans.cluster_centers_) # 10*20
        codewords_A.append(subspace_centroids_A)

    # Convert lists to tensors
    subspaces_A = torch.stack(subspaces_A, dim=0)
    codewords_A = torch.stack(codewords_A, dim=0) # 5*10*20

    print("Subspaces A shape:", subspaces_A.shape) # torch.Size([5, 500, 20])
    print("Codewords A shape:", codewords_A.shape) # torch.Size([5, 10, 20])


    # Initialize lists to store subspaces and centroids
    subspaces_B = []
    codewords_B = []

    # Create subspaces and centroids
    for k in range(m):
        subspace_B = B[k * CB : (k + 1) * CB, :]  # Extract the subspace along x-axis
        subspaces_B.append(subspace_B)

        # Apply KMeans on the column vectors within the subspace
        kmeans = KMeans(n_clusters=lB, n_init=10)
        kmeans.fit(subspace_B.T)  # Transpose to cluster along columns (column vectors)
        subspace_centroids_B = torch.tensor(kmeans.cluster_centers_)
        codewords_B.append(subspace_centroids_B)

    # Convert lists to tensors
    subspaces_B = torch.stack(subspaces_B, dim=0)
    codewords_B = torch.stack(codewords_B, dim=0)

    print("Subspaces B shape:", subspaces_B.shape)  # torch.Size([5, 20, 300])
    print("Codewords B shape:", codewords_B.shape)  # torch.Size([5, 6, 20])

    # Sample precomputed codewords for A and B (You should replace these with your actual codewords)
    lookup_table = torch.zeros((m,lA,lB))
    for i in range(m):
        for j in range(lA):
            for k in range(lB):
                lookup_table[i][j][k] = torch.matmul(codewords_A[i][j], codewords_B[i][k]) # for each subspace, get catersian product of A,B codeword

    print("lookup_table.shape: ", lookup_table.shape)

    return lA, lB, codewords_A, codewords_B, lookup_table

# used to save codeword and lookup_table to pt
class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key,value in tensor_dict.items():
            setattr(self, key, value)

datasetDir = '/home/shuhao/Downloads/AMMBench/build/benchmark/torchscripts/VQ/MtxPt'
saveDir = '/home/shuhao/Downloads/AMMBench/build/benchmark/torchscripts/VQ/CodewordLookUpTable'

for datasetName in ["GIST1M"]:
    for m in [1, 10]: # Number of subspaces

        # load
        A = torch.load(f'{datasetDir}/{datasetName}_A.pt')
        B = torch.load(f'{datasetDir}/{datasetName}_B.pt')

        # calculate codeword and lookup_table
        lA, lB, codewords_A, codewords_B, lookup_table = getCodewordAndLookUpTable(A, B, m)

        # calculate error
        C=A # actually C,D are testing matrices, and usually different from training matrices A,B. but its ok to make them the same also, cuz we more focus on latency
        D=B
        relativeFroError = "notCalculated" #quantize(C, D, codewords_A, codewords_B, lookup_table)

        # save codeword and lookup_table
        tensor_dict = {
            # 'A': A,
            # 'B': B,
            'codewordsA': codewords_A,
            'codewordsB': codewords_B,
            'lookUpTable': lookup_table,
            'datasetName': datasetName,
            'm': m,
            'lA': lA,
            'lB': lB,
            'relativeFroError': relativeFroError
        }
        tensors = TensorContainer(tensor_dict)
        tensors = torch.jit.script(tensors)
        tensors.save(f'{saveDir}/{datasetName}_m{m}_lA{lA}_lB{lB}.pth')
        print(datasetName, m, lA, lB, relativeFroError)