package org.sciscala.ndscala

object Bench extends App{

import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Random
import ONNXScalaOps._

val thisRandom = new Random(42)
val iters = 5
val lr = 0.000001f

val lrs:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = Tensor(Array.fill(10000)(lr), 10000,1)
val moreLrs:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] = Tensor(Array.fill(100000000)(lr), 10000,10000)

val ones:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = Tensor(Array.fill(10000)(1.0f), 10000,1)
val moreOnes:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] = Tensor(Array.fill(100000000)(1.0f), 10000,10000)
//For scaling
val some10ks:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = Tensor(Array.fill(10000)(10000.0f), 10000,1)
val more10ks:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] = Tensor(Array.fill(100000000)(10000.0f), 10000,10000)

//TODO: check without type annotations when it's working
val arrX:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrY:Array[Float] = (Array.fill(10000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW0:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW1:Array[Float] = (Array.fill(10000)(thisRandom.nextFloat)).map(_.toFloat)

val x:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] = Tensor(arrX,10000,10000)
val y:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]]  = Tensor(arrY, 10000,1)

//TODO: call recursively
var w0:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] = (Tensor(arrW0, 10000,10000) - moreOnes) / more10ks
var w1:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = (Tensor(arrW1, 10000,1) - ones ) /some10ks

def train = {
//     val future = async {
      val l1:Tensor[Float, Mat[?,?,?,MatShape[10000,10000]]] =  (x matmul w0).sigmoid() // one / ((-(x dot w0)).exp() + one)
      val l2:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = (l1 matmul w1).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val error = y - l2
      println("error: " + error.data.sum)
      val l2Delta:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = (error) * (l2 * (ones - l2))
      val l1Delta =  (l2Delta matmul w1.transpose) * (l1 * (moreOnes - l1))

      //Simulate in-place += op here 
      w1 = w1 + (((l1.transpose) matmul l2Delta)*lrs)
//      println("w updt" + ((l1.transpose) matmul l2Delta).data(50))
      w0 = w0 + (((x.transpose) matmul l1Delta)*moreLrs)
}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
