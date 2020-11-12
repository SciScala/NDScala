package org.sciscala.ndscala
import scala.language.implicitConversions

import org.platanios.tensorflow.api._
import ai.djl._
import ai.djl.ndarray._
import org.emergentorder.onnx.Tensors._

import TensorFlowOps._
import ONNXScalaOps._
import DJLOps._

object Mix extends App{
val a: TFTensor[Int, Mat[?,?,?,MatShape[1,2]]] = (Array(1,2), Mat(1,2)) 
val b: org.emergentorder.onnx.Tensors.Tensor[Int, Mat[?,?,?,MatShape[1,2]]] = org.emergentorder.onnx.Tensors.Tensor(Array(42, 84), 1,2)
val c: DJLNDArray[Int, Mat[?,?,?,MatShape[1,2]]] = (Array(3,5), Mat(1,2)) 

val d1: TFTensor[Int, Mat[?,?,?,MatShape[1,2]]] = a + b
val d2: org.emergentorder.onnx.Tensors.Tensor[Int, Mat[?,?,?,MatShape[1,2]]] = b + a
val arr = (d1 === d2)

println("TF and OS tensors are same: " + arr.getElementAtFlattenedIndex(1))

println(d1._1(0) + " result ")

//need to explicitly convert here because it requires 2 conversions, which won't chain automatically
//OSTensor is the intermediate here, so it only needs 1 conversion
val e: TFTensor[Int, Mat[?,?,?,MatShape[1,2]]] = a + toTFTensor(c)
println(e._1(0) + " result ")

val f: DJLNDArray[Int, Mat[?,?,?,MatShape[1,2]]] = ((Array(3,5), Mat(1,2))) + b
println(f._1(0).toString + " result ")

}

