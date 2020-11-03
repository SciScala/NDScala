package org.sciscala.ndscala

object Bench extends App{

import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import org.platanios.tensorflow.api._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

import TensorFlowOps._

//TODO: Don't use TF directly here
val iters = 5

val arrX:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrY:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)
val arrW0:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrW1:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)

val x: Tensor[Float] = (arrX, Array(10000,10000))
val y: Tensor[Float]  = (arrY, Array(10000,1))

//TODO: call recursively

val w0: Tensor[Float] = (arrW0, Array(10000,10000))
val w1: Tensor[Float] = (arrW1, Array(10000,1))

val one: Tensor[Float] = (Array(1.0f), Array(1))

def train = {
//     val future = async {

      val l1: Tensor[Float] =  ((x matmul w0)).sigmoid // one / ((-(x dot w0)).exp() + one)
      val l2: Tensor[Float] = (l1 matmul w1).sigmoid // one / ((-(l1 dot w1)).exp() + one)

      val l2Delta: Tensor[Float] = (y - l2) * (l2 * (one - l2))
      val l1Delta: Tensor[Float] =  ((l2Delta matmul w1.transpose()):Tensor[Float]) * (l1 * (one - l1))


  //Simulate in-place += op here

      
      println(w1 + ((l1.transpose()) matmul l2Delta) )
      println(w0 + ((x.transpose()) matmul l1Delta))
}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
