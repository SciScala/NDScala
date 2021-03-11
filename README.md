Training a (shape-safe) neural network in 11 lines:

In NDScala:
```scala
//Declaring types and their corresponding values //TODO - derive values from the types here
type Mat10kX10k = 10000 #: 10000 #:SNil
type AxisLabels = "AxisLabel" ##: "AxisLabel" ##: TSNil
val mat10kX10k = 10000 #: 10000 #:SNil
val axisLabels = "AxisLabel" ##: "AxisLabel" ##: TSNil

//Some setup
val ones = Tensor(Array.fill(100000000)(1.0f),"TensorLabel",axisLabels, mat10kX10k)
def arrW0:Array[Float] = ??? //Your initialized weights, layer 0, size 100m 
def arrW1:Array[Float] = ??? //Your initialized weights, layer 1, size 100m

def train(x: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          y: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          iter: Int) =
    var w0 = (Tensor(arrW0,"TensorLabel", axisLabels, mat10kX10k) - ones)
    var w1 = (Tensor(arrW1,"TensorLabel", axisLabels, mat10kX10k) - ones )
    for j <- 0 until iter
    do
        val l1 =  (x.matmul(w0)).sigmoid()
        val l2 = (l1.matmul(w1)).sigmoid()
        val error = y - l2
        val l2Delta = (error) * (l2 * (ones - l2))
        val l1Delta =  (l2Delta.matmul(w1.transpose))
        val w0New = w0 + (((x.transpose).matmul(l1Delta))) //Simulating in-place op
        val w1New = w1 + (((l1.transpose).matmul(l2Delta))) //Simulating in-place op
```

And for reference, in NumPy, in 10 lines:

```python
def train(X,Y,iter): 
    syn0 = 2*np.random.random((10000,10000)).astype('float32') - 1
    syn1 = 2*np.random.random((10000,1000)).astype('float32') - 1
    for j in range(iter): 
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))  
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1)))) 
        error = y - l2
        l2_delta = (error)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta) 
```

Which runs ~2x slower than the NDScala version (using NumPy w/ MKL).
