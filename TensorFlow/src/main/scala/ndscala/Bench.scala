package org.sciscala.ndscala

object Bench extends App{

import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import org.platanios.tensorflow.api._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import org.emergentorder.onnx.Tensors._

import TensorFlowOps._

val iters = 5

val arrX:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrY:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)
val arrW0:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrW1:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)

val x: TFTensor[Float,Mat[?,?,?,MatShape[10000,10000]]] = (arrX, Mat(10000,10000))
val y: TFTensor[Float,Mat[?,?,?,MatShape[10000,1]]]  = (arrY, Mat(10000,1))

//TODO: call recursively

val w0: TFTensor[Float,Mat[?,?,?,MatShape[10000,10000]]] = (arrW0, Mat(10000,10000))
val w1: TFTensor[Float,Mat[?,?,?,MatShape[10000,1]]] = (arrW1, Mat(10000,1))

val one: TFTensor[Float,Vec[?,?, VecShape[1]]] = (Array(1.0f), Vec(1))

def train = {
//     val future = async {

      val l1: TFTensor[Float,Mat[?,?,?,MatShape[10000,10000]]] =  ((x matmul w0)).sigmoid // one / ((-(x dot w0)).exp() + one)
      val l2: TFTensor[Float,Mat[?,?,?,MatShape[10000,1]]] = (l1 matmul w1).sigmoid // one / ((-(l1 dot w1)).exp() + one)

      val l2Delta: TFTensor[Float,Mat[?,?,?,MatShape[10000,1]]] = (y - l2) * (l2 * (one - l2))
      val l1Delta: TFTensor[Float,Mat[?,?,?,MatShape[10000,10000]]] =  ((l2Delta matmul w1.transpose())) * (l1 * (one - l1))


  //Simulate in-place += op here

      
      println(w1 + ((l1.transpose()) matmul l2Delta) )
      println(w0 + ((x.transpose()) matmul l1Delta))
}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
