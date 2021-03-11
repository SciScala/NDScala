Training a (shape-safe) neural network in 11 lines:

In NDScala:
```scala
//Some setup - to be hidden
val thisRandom = new Random(42)
val ones = Tensor(Array.fill(10000000)(1.0f),"TensorLabel", "AxisLabel" ##: "AxisLabel" ##: TSNil, 10000 #: 1000 #: SNil)
val moreOnes = Tensor(Array.fill(100000000)(1.0f),"TensorLabel","AxisLabel" ##: "AxisLabel" ##: TSNil, 10000 #: 10000 #: SNil)
val arrW0:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW1:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)

def train(x: Tensor[Float, ("TensorLabel", "AxisLabel" ##: "AxisLabel" ##: TSNil, 10000 #: 10000 #:SNil)],
          y: Tensor[Float, ("TensorLabel", "AxisLabel" ##: "AxisLabel" ##:  TSNil, 10000 #: 1000 #:SNil)],
          iter: Int) =
    var w0 = (Tensor(arrW0,"TensorLabel","AxisLabel" ##: "AxisLabel" ##: TSNil, 10000 #: 10000 #: SNil) - moreOnes)
    var w1 = (Tensor(arrW1,"TensorLabel","AxisLabel" ##: "AxisLabel" ##: TSNil, 10000 #: 1000 #: SNil) - ones )
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
