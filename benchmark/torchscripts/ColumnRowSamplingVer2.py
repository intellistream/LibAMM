import torch
import time	
import os

def get_first_element(tensor):
	if tensor.numel() == 1:
		return tensor.item()
	else:
		return tensor[0].item()
	
def is_empty_tensor(tensor):
	return tensor.numel() == 0

@torch.jit.script
def CRS(A: torch.Tensor, B: torch.Tensor, k: int):
	# Get the dimension of A
	n, m = A.shape
	
	assert m == B.shape[1]
	assert k < m
	
	# probability distribution
	# dist = torch.distributions.Uniform(0, 1) # default: uniform

	
	# sample k indices from range 0 to m for given probability distribution
	# sample = dist.sample((n,))
	sample = torch.rand(n)
	sample = torch.div(sample, sample.sum())
	D = torch.diag(1.0 / torch.sqrt(k * sample))
	
	
	column_indices = torch.multinomial(sample, k, replacement=False)
	S = torch.zeros(k, n)
	for row, col in enumerate(column_indices):
		S[row, col] = 1
	
	a = torch.matmul(torch.matmul(A.t(), D), S.t())
	b = torch.matmul(torch.matmul(a, S), D)
	
	
	return torch.matmul(b, B)


def main():
	
	
	width = 5000
	A = torch.rand(width, 5000)
	B = torch.rand(width, 5000)
	
	t = time.time()
	
	aResult = CRS(A, B, 500)
	print("approximate: " + str(time.time() - t) + "s")
	
	print(aResult)
	
	# exact result
	t = time.time()
	eResult = torch.matmul(A.t(), B)
	print("\nExact: " + str(time.time() - t) + "s")
	
	print(eResult)
	
	print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))
	
	script = CRS.save("CRSV2.pt")

if __name__ == '__main__':
	main()
	