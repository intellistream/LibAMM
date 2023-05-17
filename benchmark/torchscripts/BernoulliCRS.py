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
def BernoulliCRS(A: torch.Tensor, B: torch.Tensor, k: int):
	# Get the dimension of A
	n, m = A.shape
	
	assert m == B.shape[1]
	assert k < m
	
	# probability distribution
	sample = torch.rand(n)							# default: uniform
	sample = torch.div(sample, sample.sum() / k) 	# sum = k as per the paper
	
	# diagonal scaling matrix P (nxn)
	P = torch.diag(1.0 / torch.sqrt(sample))
	
	# random diagonal sampling matrix K (nxn)
	sample = (torch.rand(n) < sample).float()
	K = torch.diag(sample) 
	
	a = torch.matmul(torch.matmul(A.t(), P), K)
	b = torch.matmul(torch.matmul(a, K), P)
	
	return torch.matmul(b, B)


def main():
	
	
	width = 1000
	A = torch.rand(width, 2000)
	B = torch.rand(width, 2000)
	
	t = time.time()
	
	aResult = BernoulliCRS(A, B, 500)
	print("approximate: " + str(time.time() - t) + "s")
	
	print(aResult)
	
	# exact result
	t = time.time()
	eResult = torch.matmul(A.t(), B)
	print("\nExact: " + str(time.time() - t) + "s")
	
	print(eResult)
	
	print("\nerror: " + str(torch.norm(aResult - eResult, p='fro').item()))
	
	script = BernoulliCRS.save("BernoulliCRS.pt")

if __name__ == '__main__':
	main()
	