import torch
import time
import os
import math

@torch.jit.script
def EWS(A: torch.Tensor, B: torch.Tensor, k: int):
	# Get the dimension of A
	m, n = A.shape
	
	assert n == B.shape[0]
	assert k < n
	
	p = B.shape[1]
	
	# probability distribution
	probs = torch.rand(m, n)

	# S matrix that samples A with scaling 
	mask = torch.rand_like(probs) < probs
	S = torch.zeros_like(A)
	S[mask] = A[mask] / probs[mask]

	# R matrix that samples B with scaling
	probs = torch.rand(n, p) 		# a diffrent probabilistic distribution
	mask = torch.rand_like(probs) < probs
	R = torch.zeros_like(B)
	R[mask] = B[mask] / probs[mask]

	return torch.matmul(S, R)


def main():
	
	width = 2000
	A = torch.rand(5000, width)
	B = torch.rand(width, 5000)

	
	t = time.time()
	
	aResult = EWS(A, B, 500)
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
	
	FDAMM_script = EWS.save("EWS.pt")

if __name__ == '__main__':
	main()
	