import numpy as np
import time

X = np.random.random((10000,10000)).astype('float32') 
y = np.random.random((1,10000)).T.astype('float32')

print(X.shape)
print(X.dtype)
iter = 50

def elevenlines(): 
    syn0 = 2*np.random.random((10000,10000)).astype('float32') - 1
    syn1 = 2*np.random.random((10000,1)).astype('float32') - 1
    tic = time.time()
    for j in range(iter): 
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))  
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1)))) 
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta)
    toc = time.time()
    print(toc-tic)

results = elevenlines() 
