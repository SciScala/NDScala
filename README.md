Training a (shape-safe) neural network in 10 lines:

In NDScala:
```scala
//After some setup
//Declaring types and their corresponding values
type Mat10kX10k = 10000 #: 10000 #:SNil
type AxisLabels = "AxisLabel" ##: "AxisLabel" ##: TSNil
val mat10kX10k = shapeOf[Mat10kX10k]
val axisLabels = tensorShapeDenotationOf[AxisLabels]

val ones = Tensor(Array.fill(100000000)(1.0f),"TensorLabel",axisLabels, mat10kX10k)

def train(x: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          y: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          w0: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          w1: Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
          iter: Int): Tuple2[Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)],
                             Tensor[Float, ("TensorLabel", AxisLabels, Mat10kX10k)]] =
    if iter == 0 then (w0, w1)
    else
        val l1 =  (x.matmul(w0)).sigmoid()
        val l2 = (l1.matmul(w1)).sigmoid()
        val error = y - l2
        val l2Delta = (error) * (l2 * (ones - l2))
        val l1Delta =  (l2Delta.matmul(w1.transpose))
        val w1New = w1 + (((l1.transpose).matmul(l2Delta)))
        val w0New = w0 + (((x.transpose).matmul(l1Delta)))
        train(x,y,w0New,w1New,iter-1)
```

and then you can fuse the operations into a single optimized ONNX graph, and execute it:

```scala
  val fusedTraining = fuseOps
  val onnxBytesTraining = fusedTraining.toByteArray
  val fusedModelTraining = new ORTModelBackend(onnxBytesTraining)

  val trainOut = fusedModelTraining.fullModel[Float, "TensorLabel", AxisLabels, Mat10kx10k](x, y, w0, w1)
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

The run time of the eager NDScala version is ~80% of that of NumPy w/MKL,
while if we fuse the ops to a graph it's just ~65%.

The PyTorch equivalent is slightly faster, at ~85% of the fused NDScala version run time.
This can be accounted for by the copy overhead of passing data between the JVM and native memory.
