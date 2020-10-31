package org.sciscala.ndscala
import scala.language.implicitConversions

import org.platanios.tensorflow.api._
import ai.djl._
import ai.djl.ndarray._

import TensorFlowOps._
import ONNXScalaOps._
import DJLOps._

object Mix extends App{
val a: org.platanios.tensorflow.api.Tensor[Int] = Tensor[Int](1, 2)
val b: org.emergentorder.onnx.Tensors.Tensor[Int] = (Array(42, 84), Array(1,2))
val c: DJLNDArray[Int] = (Array(3,5), Array(1,2)) 

val d1: org.platanios.tensorflow.api.Tensor[Int] = a + b
val d2: org.emergentorder.onnx.Tensors.Tensor[Int] = a + b
val arr = (d1 === d2)

println("TF and OS tensors are same: " + arr.getElementAtFlattenedIndex(1))

println(d1._1(0) + " result ")


//TODO: fix need to explicitly convert here
val e: Tensor[Int] = a + toTFTensor(fromDJLNDArray(c))
println(e._1(0) + " result ")

//Needs the type hint here because TF ops was imported first, defaults to that
val f: DJLNDArray[Int] = ((Array(3,5), Array(1,2)): DJLNDArray[Int]) + b
println(f._1(0).toString + " result ")

}

