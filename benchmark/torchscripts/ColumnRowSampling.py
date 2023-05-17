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
	A = A.t()
	n, m = A.shape
	
	assert n == B.shape[0]
	assert k < m
	
	Ai = torch.linalg.norm(A, dim=0)
	Aj = torch.linalg.norm(A, dim=1)
	Bi = torch.linalg.norm(B, dim=0)
	Bj = torch.linalg.norm(B, dim=1)
	
	print(Ai.shape)
	print(Aj.shape)
	print(Bi.shape)
	print(Bj.shape)
	
	dot_product1 = torch.dot(Aj, Bj)
	print(dot_product1)
	
	
	# probability distribution
	probs = torch.ones(n) / n # default: uniform
		
	# sample k indices from range 0 to n for given probability distribution
	indices = torch.multinomial(probs, k, replacement=False)

	# Sample k columns from A
	A_sampled = A[indices, :]
	A_sampled = torch.div((A_sampled / k).t(), probs[::2])

	# Sample k rows from B
	B_sampled = B[indices, :]
	
	# Compute the matrix product
	result = A_sampled.matmul(B_sampled)
	return result


def main():
	
	width = 1000
	A = torch.rand(5000, width)
	B = torch.rand(width, 5000)

	
	t = time.time()
	
	aResult = CRS(A, B, 500)
	print("approximate: " + str(time.time() - t) + "s")
	
	print(aResult)
	
	# exact result
	t = time.time()
	eResult = torch.matmul(A, B)
	print("\nExact: " + str(time.time() - t) + "s")
	
	print(eResult)
	
	print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))
	
	# FDAMM_script = CRS.save("CRS.pt")

if __name__ == '__main__':
	main()
	