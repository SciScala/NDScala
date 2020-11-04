package org.sciscala.ndscala

object Bench extends App{

import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import ai.djl._
import ai.djl.ndarray._
import org.emergentorder.onnx.Tensors._

import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

import DJLOps._

val iters = 5

val arrX:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrY:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)
val arrW0:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrW1:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)

val x: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]] = (arrX, Mat(10000,10000))
val y: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]  = (arrY, Mat(10000,1))

//TODO: call recursively

val w0: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]] = (arrW0, Mat(10000,10000))
val w1: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]] = (arrW1, Mat(10000,1))

//val one: DJLNDArray[Float, Vec[?,?,1,VecShape[1]]] = (Array(1.0f), Vec(1))
val ones:Tensor[Float, Mat[?,?,?,MatShape[10000,1]]] = Tensor(Array.fill(10000)(1.0f), 10000,1)

def train = {
//     val future = async {

      val l1: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]] =  ((x:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) matmul (w0:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]])).sigmoid() // one / ((-(x dot w0)).exp() + one)
 
      val l2: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]] = ((l1:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) matmul (w1:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]])).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val l2Delta: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]] = ((y: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) - (l2: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]])) * ((l2:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) * ((ones:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) - (l2:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]])))
      val l1Delta: DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]] =  ((l2Delta:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) matmul (w1.transpose:DJLNDArray[Float, Mat[?,?,?,MatShape[1,10000]]])) * ((l1:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) * ((ones:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) - (l1:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]])))


  //Simulate in-place += op here

      //println to force it to block here
      println((w1:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]) + ((l1.transpose:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) matmul (l2Delta:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,1]]]))) 
      println((w0:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) + ((x.transpose:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]]) matmul (l1Delta:DJLNDArray[Float, Mat[?,?,?,MatShape[10000,10000]]])))

}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
