package org.sciscala.ndscala 

import scala.util.Random
import scala.reflect.ClassTag
import scala.language.implicitConversions
import spire.random.Dist
import spire.math.Numeric
//import spire.implicits._
import org.sciscala.ndscala.union._
//import org.platanios.tensorflow.api.tensors._
//import org.platanios.tensorflow.api.core.types._
//import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api._
//import org.platanios.tensorflow.api.tensors.ops.Math._ 

//TODO: fix for shape safety
object TensorFlowOps {
//  implicit def convert[DType: ClassTag: TF](d: DType): Tensor[DType] = Tensor(d) 
  implicit def toTFTensor[DType: ClassTag: TF](t: (Array[DType], Array[Int])): Tensor[DType] = Tensor(t._1).reshape(Shape(t._2: _*))
  implicit def fromTFTensor[DType: ClassTag: TF](t: Tensor[DType]): (Array[DType], Array[Int]) = (Array(t.toArray: _*) , Array(t.shape.toArray: _*)) 
}


//WTF infinite recursion
given NDArrayOps[Tensor]{
//  import TensorFlowOps._

//import org.platanios.tensorflow.api.tensors.ops.Math._ 
//  implicit def toTensor[DType: ClassTag: TF](t: (Array[DType], Array[Int])): Tensor[DType] = Tensor(t._1).reshape(Shape(t._2: _*))
//  implicit def fromTensor[DType: ClassTag: TF](t: Tensor[DType]): (Array[DType], Array[Int]) = (Array(t.toArray: _*) , Array(t.shape.toArray: _*)) 
  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): Tensor[DType] = ??? //broken: needs tf type in sig Tensor.zeros[DType](Shape(shape: _*))
  def ones[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): Tensor[DType] = ??? //broken: Tensor.ones[DType](Shape(shape: _*)) 
  def full[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int], value: DType): Tensor[DType] = ??? //broken: Tensor.fill(Shape(shape: _*))(value)

  
  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): Tensor[DType] = ???

  //Unary ops
  //caution: infinite
  extension[DType <: Supported : ClassTag : IsSupported](arr: Tensor[DType]) def reShape(newShape: Array[Int]): Tensor[DType] = arr.reshape(Shape(newShape).asArray)
  extension[DType <: Supported : ClassTag : IsSupported](arr: Tensor[DType]) def transpose: Tensor[DType] = arr.transpose()
  //caution: infinite
  extension[DType <: Supported : ClassTag : IsSupported](arr: Tensor[DType]) def transpose(axes: Array[Int], dummy: Option[Boolean]): Tensor[DType] = arr.transpose(Shape(axes: _*))
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]) def round(): Tensor[DType]  = arr.round
  //Top-level slice only
  extension[DType <: Supported : ClassTag: IsSupported](arr: Tensor[DType]) def slice(start: Int, end: Int, dummy: Option[Boolean]): Tensor[DType] = arr(start :: end)

  //Caution: infinite

  extension[DType <: Supported : ClassTag : IsSupported](arr: Tensor[DType]) def squeeze: Tensor[DType] = arr.squeeze(Array(0)) 
  extension[DType <: Supported : ClassTag: IsSupported](arr: Tensor[DType]) def squeeze(index: Array[Int], dummy: Option[Boolean]): Tensor[DType] =  arr.squeeze(index)

  extension[DType <: Supported : ClassTag: IsSupported](arr: Tensor[DType]) def rank: Int = arr.rank
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], min: DType, max: DType): Tensor[DType] = arr.clip(min, max) 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def unary_- : Tensor[DType] = - arr 

//  def concat[DType <: Supported : ClassTag : IsSupported](axis: Int, arr: Seq[Tensor[DType]]): Tensor[DType] = ???
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[Tensor[DType]]): Tensor[DType] = ???

  //Below calls not valid, infinite recursion.. need to specialize?
  //Also, time to get rid of classtag"?
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported](arr: Tensor[DType]) def abs(): Tensor[DType] = arr.abs
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def ceil(): Tensor[DType] = arr.ceil
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def floor(): Tensor[DType] = arr.floor 

  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def log(): Tensor[DType]= arr.log 
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def exp(): Tensor[DType] = arr.exp
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def sqrt(): Tensor[DType] = arr.sqrt
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def cos(): Tensor[DType] = arr.cos
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def cosh(): Tensor[DType] = arr.cosh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def sin(): Tensor[DType] = arr.sin
  extension[DType <:FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def sinh(): Tensor[DType] = arr.sinh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def tan(): Tensor[DType] = arr.tan
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def tanh(): Tensor[DType] = arr.tanh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def acos(): Tensor[DType] = arr.acos
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def acosh(): Tensor[DType] = arr.acosh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def asin(): Tensor[DType] = arr.asin
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def asinh(): Tensor[DType] = arr.asinh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def atan(): Tensor[DType] = arr.atan
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported](arr: Tensor[DType]) def atanh(): Tensor[DType] = arr.atanh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def sigmoid(): Tensor[DType] = ???
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def relu(): Tensor[DType] = ???

  //Binary Tensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def + (other: Tensor[DType]): Tensor[DType] = arr + other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def - (other: Tensor[DType]): Tensor[DType] = arr - other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def * (other: Tensor[DType]): Tensor[DType] = arr * other 
  extension[DType <: NumericSupported :  ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def ** (other: Tensor[DType]): Tensor[DType] = arr ** other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def / (other: Tensor[DType]): Tensor[DType] = arr / other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def % (other: Tensor[DType]): Tensor[DType] = arr % other 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def > (other: Tensor[DType]): Tensor[Boolean] = arr > other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def >= (other: Tensor[DType]): Tensor[Boolean] = arr >= other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def < (other: Tensor[DType]): Tensor[Boolean] = arr < other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def <= (other: Tensor[DType]): Tensor[Boolean] = arr <= other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def ==== (other: Tensor[DType]): Tensor[Boolean] = arr === other 
    // arr === other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def !=== (other: Tensor[DType]): Tensor[Boolean] = arr =!= other 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def max (d: Tensor[DType]): Tensor[DType] = arr maximum d 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def min (d: Tensor[DType]): Tensor[DType] = arr minimum d 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](arr: Tensor[DType]) def matmul (other: Tensor[DType]): Tensor[DType] = arr tensorDot (other, arr.rank)
}
