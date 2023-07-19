import torch
import time

@torch.jit.script
def SRHT(A: torch.Tensor, B: torch.Tensor, m: int):
	# Get the dimension of A
	A = A.t()
	assert A.shape[0] == B.shape[0]
	n = A.shape[0]
	
	# a diagonal matrix D with entries either -1 or 1
	diag_elements = torch.randint(2, (n,), dtype=torch.float32) * 2 - 1
	D = torch.diag(diag_elements)

	# unnormalized Hadamard transform matrix H
	l = int(2 ** int(torch.ceil(torch.log2(torch.tensor(n)))))
	H = torch.empty(l, l)

	for i in range(l):
		for j in range(l):
			H[i, j] = (-1) ** (bin(i & j).count('1') % 2)
	H = H[:n, :n]

	# Random subsampling matrix S
	S = torch.zeros((m, n))
	for i in range(m):
		idx = torch.randint(n, (1,)).item()
		S[i, int(idx)] = 1

	Pi = (1 / torch.sqrt(torch.tensor(m).float())) * torch.matmul(torch.matmul(S, H), D)
	A_transform = torch.matmul(Pi, A)
	B_transform = torch.matmul(Pi, B)
	return torch.matmul(A_transform.t(), B_transform)

def main():
	
	width = 500
	A = torch.rand(1000, width)
	B = torch.rand(width, 1000)

	
	t = time.time()
	
	aResult = SRHT(A, B, 100)
	print("approximate: " + str(time.time() - t) + "s")
	
	print(aResult)
	
	# exact result
	t = time.time()
	eResult = torch.matmul(A, B)
	print("\nExact: " + str(time.time() - t) + "s")
	
	print(eResult)

	difference = aResult - eResult
	print("\nFrobenius norm error: " + str(torch.linalg.norm(difference, ord='fro').item()))
	print("\nSpectral norm bound: " + str(torch.linalg.norm(difference, ord=2).item()))
	
	script = SRHT.save("SRHT.pt")

if __name__ == '__main__':
	main()
	