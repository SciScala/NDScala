import torch
import time
#import mkl
#mkl.set_num_threads(4)

X = torch.randn((10000,10000)).float() 
y = torch.randn((1000,10000)).T.float()

print(X.shape)
print(X.dtype)
iter = 1

def elevenlines(): 
    syn0 = 2*torch.randn((10000,10000)).float() - 1
    syn1 = 2*torch.randn((10000,1000)).float() - 1
    tic = time.time()
    for j in range(iter):
        l1 = 1/(1+torch.exp(-(torch.matmul(X,syn0))))
        l2 = 1/(1+torch.exp(-(torch.matmul(l1,syn1)))) 
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.matmul(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.matmul(l2_delta)
        syn0 += X.T.matmul(l1_delta)
    toc = time.time()
    print(toc-tic)

results = elevenlines() 
