package org.sciscala.ndscala

object Bench extends App{

import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import ai.djl._
import ai.djl.ndarray._

import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

import DJLOps._

val iters = 5

val arrX:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrY:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)
val arrW0:Array[Float] = ((0 until 100000000).toArray).map(_.toFloat)
val arrW1:Array[Float] = ((0 until 10000).toArray).map(_.toFloat)

val x: DJLNDArray[Float] = (arrX, Array(10000,10000))
val y: DJLNDArray[Float]  = (arrY, Array(10000,1))

//TODO: call recursively

val w0: DJLNDArray[Float] = (arrW0, Array(10000,10000))
val w1: DJLNDArray[Float] = (arrW1, Array(10000,1))

val one: DJLNDArray[Float] = (Array(1.0f), Array(1))

def train = {
//     val future = async {

      val l1: DJLNDArray[Float] =  ((x:DJLNDArray[Float]) matmul (w0:DJLNDArray[Float])).sigmoid() // one / ((-(x dot w0)).exp() + one)
      val l2: DJLNDArray[Float] = ((l1:DJLNDArray[Float]) matmul (w1:DJLNDArray[Float])).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val l2Delta: DJLNDArray[Float] = ((y: DJLNDArray[Float]) - (l2: DJLNDArray[Float])) * ((l2:DJLNDArray[Float]) * ((one:DJLNDArray[Float]) - (l2:DJLNDArray[Float])))
      val l1Delta: DJLNDArray[Float] =  ((l2Delta:DJLNDArray[Float]) matmul (w1.transpose:DJLNDArray[Float])) * ((l1:DJLNDArray[Float]) * ((one:DJLNDArray[Float]) - (l1:DJLNDArray[Float])))


  //Simulate in-place += op here

      //println to force it to block here
      println(w1 + ((l1.transpose:DJLNDArray[Float]) matmul (l2Delta:DJLNDArray[Float]))) 
      println(w0 + ((x.transpose:DJLNDArray[Float]) matmul (l1Delta:DJLNDArray[Float])))
}

val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
