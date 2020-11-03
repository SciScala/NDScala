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

val x: DJLNDArray[Float, Mat[10000,10000,MatShape[10000,10000]]] = (arrX, Mat(10000,10000))
val y: DJLNDArray[Float, Mat[10000,1,MatShape[10000,1]]]  = (arrY, Mat(10000,1))

//TODO: call recursively

val w0: DJLNDArray[Float, Mat[10000,10000,MatShape[10000,10000]]] = (arrW0, Mat(10000,10000))
val w1: DJLNDArray[Float, Mat[10000,1,MatShape[10000,1]]] = (arrW1, Mat(10000,1))

val one: DJLNDArray[Float, Vec[1,VecShape[1]]] = (Array(1.0f), Vec(1))

def train = {
//     val future = async {

      val l1: DJLNDArray[Float, Axes] =  ((x:DJLNDArray[Float, Axes]) matmul (w0:DJLNDArray[Float, Axes])).sigmoid() // one / ((-(x dot w0)).exp() + one)
      val l2: DJLNDArray[Float, Axes] = ((l1:DJLNDArray[Float, Axes]) matmul (w1:DJLNDArray[Float, Axes])).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val l2Delta: DJLNDArray[Float, Axes] = ((y: DJLNDArray[Float, Axes]) - (l2: DJLNDArray[Float, Axes])) * ((l2:DJLNDArray[Float, Axes]) * ((one:DJLNDArray[Float, Axes]) - (l2:DJLNDArray[Float, Axes])))
      val l1Delta: DJLNDArray[Float, Axes] =  ((l2Delta:DJLNDArray[Float, Axes]) matmul (w1.transpose:DJLNDArray[Float, Axes])) * ((l1:DJLNDArray[Float, Axes]) * ((one:DJLNDArray[Float, Axes]) - (l1:DJLNDArray[Float, Axes])))


  //Simulate in-place += op here

      //println to force it to block here
      println(w1 + ((l1.transpose:DJLNDArray[Float, Axes]) matmul (l2Delta:DJLNDArray[Float, Axes]))) 
      println(w0 + ((x.transpose:DJLNDArray[Float, Axes]) matmul (l1Delta:DJLNDArray[Float, Axes])))
}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
