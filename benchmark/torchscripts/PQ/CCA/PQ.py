import torch

import time


def kmeans(X, K, max_iters=100):
    N, D = X.shape

    # Randomly initialize centroids
    centroids = X[torch.randperm(N)[:K]]

    for _ in range(max_iters):
        # Calculate distances between data points and centroids
        distances = torch.cdist(X, centroids)

        # Assign data points to the nearest centroid
        labels = torch.argmin(distances, dim=1)

        # Update centroids
        for k in range(K):
            if torch.sum(labels == k) > 0:
                centroids[k] = torch.mean(X[labels == k], dim=0)

    return centroids, labels


class MyModule(torch.nn.Module):
    def __init__(self, prototypes):
        super(MyModule, self).__init__()
        # self.fc = torch.nn.Linear(10, 5)
        self.prototypes = torch.nn.Parameter(prototypes)
        self.QA = torch.nn.Parameter(torch.rand(5, 5))
        # self.register_parameter("prototypes", self.prototypes)

    def forward(self, A: torch.Tensor):
        return torch.matmul(A, A.T) + torch.sum(self.prototypes[0]) + torch.sum(self.QA)


def save_model(model, path, X):
    tx = X.to('cpu')
    # tx=X
    model2 = model.to('cpu')
    # model2=model
    # model2.eval()
    X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    traced_model = torch.jit.script(model2)
    ru = traced_model(tx)

    traced_model.save(path)


def main():
    learning = True

    A = torch.load('/home/heyuhao/AMMBench/build/benchmark/datasets/CCA_MNIST_train_A_minus_mean.pt')
    B = A.t()
    N = A.shape[0]
    D = A.shape[1]
    M = B.shape[1]

    # Number of subspaces
    C = 10

    # Number of prototypes per subspace
    K = 64

    # Calculate dimension of each subspace
    D_c = D // C

    # Initialize a list to hold prototypes for each subspace
    prototypes = []

    if learning:
        print("Starting prototypes learning on training set size of "
              + str(A.shape[0]) + ", " + str(A.shape[1]))
        t = time.time()

        for c in range(C):
            # Get the indices for this subspace
            subspace_indices = range(c * D_c, (c + 1) * D_c)

            # Slice the matrix A to get the subspace
            A_subspace = A[:, subspace_indices]

            # Convert to numpy for KMeans
            A_subspace_np = A_subspace.detach().numpy()

            # Run KMeans on the subspace
            centroids_torch, LB = kmeans(A_subspace, K, 10)

            # Get the centroids (prototypes)
            # centroids = kmeans.cluster_centers_

            # Convert back to PyTorch tensor
            # centroids_torch = torch.from_numpy(centroids)

            # Append to the list of prototypes
            prototypes.append(centroids_torch)

        # Now prototypes[c] gives the K prototypes for subspace c

        print("\nPrototype Learning: " + str(time.time() - t) + "s")
        model = MyModule(torch.stack(prototypes))
        save_model(model.to('cpu'), "/home/heyuhao/AMMBench/benchmark/torchscripts/PQ/CCA/prototypes_CCA_MNIST_train_A_minus_mean_C10_K64.pt", torch.zeros(5, 5))

    else:
        print("Loading prototypes from serialized pickle file\n")
        prototypes = torch.load('prototypes.pt')

    # Initialize an empty list to store the encoded rows of A
    A_encoded = []

    total = time.time()

    for a in A:
        # Initialize an empty list to store the encodings for this row
        a_encoded = []

        for c in range(C):
            # Get the prototypes for this subspace
            prototypes_c = prototypes[c]

            # Get the subvector for this subspace
            a_subvector = a[c * D_c: (c + 1) * D_c]

            # Calculate the distances from the subvector to each prototype
            distances = torch.norm(prototypes_c - a_subvector, dim=1)

            # Find the index of the closest prototype
            closest_prototype_index = torch.argmin(distances)

            # Append this index to the encoded row
            a_encoded.append(closest_prototype_index)

        # Convert the list of encoded subvectors to a PyTorch tensor
        a_encoded_tensor = torch.tensor(a_encoded)

        # Append this encoded row to the list of all encoded rows
        A_encoded.append(a_encoded_tensor)

    # Convert the list of all encoded rows to a PyTorch tensor
    A_encoded_tensor = torch.stack(A_encoded)

    # Now A_encoded_tensor is the encoded version of A

    print("Encoding Function: " + str(time.time() - total) + "s")

    # Assume that B is a PyTorch tensor of size D x M

    # Initialize a list to store the tables for each subspace
    tables = []

    t = time.time()

    for c in range(C):
        # Get the prototypes for this subspace
        prototypes_c = prototypes[c]

        # Slice B to get the corresponding subspace
        B_subspace = B[c * D_c: (c + 1) * D_c, :]

        # Initialize an empty list to store the table for this subspace
        table_c = []

        for prototype in prototypes_c:
            # Calculate the dot product of the prototype with each column in B_subspace
            dot_products = torch.matmul(prototype, B_subspace)

            # Append the dot products to the table for this subspace
            table_c.append(dot_products)

        # Convert the table for this subspace to a PyTorch tensor
        table_c_tensor = torch.stack(table_c)

        # Append the table for this subspace to the list of all tables
        tables.append(table_c_tensor)

    # Now tables[c] gives the lookup table for subspace c

    print("Table Construction: " + str(time.time() - t) + "s")

    # Initialize a list to store the result of the approximated matrix product
    result = []

    t = time.time()

    # Iterate over the encoded rows of A
    for a_encoded in A_encoded_tensor:
        # Initialize a tensor to store the sum of the dot products for this row
        row_sum = torch.zeros(B.size(1))

        # Iterate over each subspace
        for c in range(C):
            # Get the index of the closest prototype for this subspace
            prototype_index = a_encoded[c].item()

            # Get the lookup table for this subspace
            table_c = tables[c]

            # Get the dot products for the closest prototype
            dot_products = table_c[prototype_index]

            # Add these dot products to the sum for this row
            row_sum += dot_products

        # Append the sum for this row to the list of all rows
        result.append(row_sum)

    # Convert the list of all rows to a PyTorch tensor
    result_tensor = torch.stack(result)

    print("Aggregation: " + str(time.time() - t) + "s")
    print("\nTotal: " + str(time.time() - total) + "s")

    # Now result_tensor is the approximated matrix product
    print("\nApproximate Result")
    print(result_tensor)

    print("\nExact Result")
    t = time.time()
    eResult = torch.matmul(A, B)
    print("Exact time: " + str(time.time() - t) + "s")
    print(eResult)
    print("\nrelative fro error: " + str((torch.norm(result_tensor - eResult, p='fro')/torch.norm(eResult, p='fro')).item()))
    print(len(prototypes), torch.stack(prototypes).size())


if __name__ == '__main__':
    main()
