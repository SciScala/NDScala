package org.sciscala.ndscala 

import scala.util.Random
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math.Numeric
import spire.implicits._
import org.sciscala.ndscala.union._
//import org.platanios.tensorflow.api.tensors._
//import org.platanios.tensorflow.api.core.types._
//import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api._

object TensorFlowOps {
//  implicit def convert[DType: ClassTag: TF](d: DType): Tensor[DType] = Tensor(d) 
  implicit def toTensor[DType: ClassTag: TF](t: (ArraySeq[DType], ArraySeq[Int])): Tensor[DType] = Tensor(t._1).reshape(Shape(t._2: _*))
  implicit def fromTensor[DType: ClassTag: TF](t: Tensor[DType]): (ArraySeq[DType], ArraySeq[Int]) = (ArraySeq(t.toArray: _*) , ArraySeq(t.shape.toArray: _*)) 
}

class TensorFlowOps extends NDArrayOps[Tensor]{
  import TensorFlowOps._

  //Nullary / factory ops
  def zeros[DType : ClassTag: Numeric: IsSupported: TF](shape: ArraySeq[Int]): Tensor[DType] = Tensor.zeros[DType](Shape(shape: _*))
  def ones[DType : ClassTag: Numeric: IsSupported: TF](shape: ArraySeq[Int]): Tensor[DType] = Tensor.ones[DType](Shape(shape: _*)) 
  def full[DType : ClassTag: Numeric: IsSupported: TF](shape: ArraySeq[Int], value: DType): Tensor[DType] = Tensor.fill(Shape(shape: _*))(value)

  
  //TODO: fix rand
  def rand[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): Tensor[DType] = ???

  //Unary ops
  def reshape[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], newShape: ArraySeq[Int]): Tensor[DType] = arr.reshape(newShape)
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Tensor[DType] = arr.transpose()
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], axes: ArraySeq[Int]): Tensor[DType] = arr.transpose(Shape(axes: _*))
  def round[DType : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType]  = arr.round
  //Top-level slice only
  def slice[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], start: Int, end: Int): Tensor[DType] = arr(start :: end)

  def squeeze[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], index: ArraySeq[Int]): Tensor[DType] = arr.squeeze(index)

  def rank[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Int = arr.rank
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], min: DType, max: DType): Tensor[DType] = arr.clip(min, max) 

  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]) : Tensor[DType] = - arr 

  def abs[DType: ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Tensor[DType] = arr.abs 
  def ceil[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.ceil
  def floor[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.floor 
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: Tensor[DType]*): Tensor[DType] = onnx.Concat11("concat", Some(ndArrayToTensor(arr)))))
  def log[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType]= arr.log 
  def exp[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.exp
  def sqrt[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.sqrt
  def cos[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.cos
  def cosh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.cosh
  def sin[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.sin
  def sinh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.sinh
  def tan[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.tan
  def tanh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.tanh
  def acos[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.acos
  def acosh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.acosh
  def asin[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.asin
  def asinh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.asinh
  def atan[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = arr.atan
//  def atanh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Atanh9("atanh", Some(arr))

  //Binary Tensor ops

  def +[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr + other 
  def -[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr - other
  def *[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr * other 
  def **[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr ** other 
  def /[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr / other
  def %[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr % other 

  def >[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr > other 
  def >=[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr >= other 
  def <[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr < other 
  def <=[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr <= other 
  def ===[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr === other 
  def !==[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = arr =!= other 

  def max[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], d: Tensor[DType]): Tensor[DType] = arr maximum d 
  def min[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], d: Tensor[DType]): Tensor[DType] = arr minimum d 

  def dot[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = arr tensorDot (other, arr.rank)
}
