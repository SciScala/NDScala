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
import org.emergentorder.onnx.Tensors.Axes
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.=!=
//import org.platanios.tensorflow.api.tensors.ops.Math._ 

object TensorFlowOps {
//  implicit def convert[DType: ClassTag: TF](d: DType): TFTensor[DType] = TFTensor(d) 
  implicit def toTFTensor[DType <: AllSupported : ClassTag: TF, Ax <: Axes](t: (Array[DType], Ax)): TFTensor[DType, Ax] = Tensor(t._1).reshape(Shape(t.shape: _*))
  implicit def fromTFTensor[DType <: AllSupported : ClassTag: TF, Ax <: Axes](t: TFTensor[DType, Ax]): (Array[DType], Ax) = {
    val tens = create(Array(t.toArray: _*) , Array(t.shape.toArray: _*)) 
    (tens._1, tens._2)
  }
}

type TFTensor[DType <: AllSupported, Ax <: Axes] = Tensor[DType]

given NDArrayOps[TFTensor]{
//  import TFTensorFlowOps._

//import org.platanios.tensorflow.api.tensors.ops.Math._ 
//  implicit def toTFTensor[DType: ClassTag: TF](t: (Array[DType], Array[Int])): TFTensor[DType] = TFTensor(t._1).reshape(Shape(t._2: _*))
//  implicit def fromTFTensor[DType: ClassTag: TF](t: TFTensor[DType]): (Array[DType], Array[Int]) = (Array(t.toArray: _*) , Array(t.shape.toArray: _*)) 
  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): TFTensor[DType, Axes] = ??? //broken: needs tf type in sig TFTensor.zeros[DType](Shape(shape: _*))
  def ones[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): TFTensor[DType, Axes] = ??? //broken: TFTensor.ones[DType](Shape(shape: _*)) 
  def full[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int], value: DType): TFTensor[DType, Axes] = ??? //broken: TFTensor.fill(Shape(shape: _*))(value)

  
  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): TFTensor[DType] = ???

  //Unary ops
  //caution: infinite
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def reShape(newShape: Array[Int]): TFTensor[DType, Bx] = arr.reshape(Shape(newShape).asArray)
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def transpose: TFTensor[DType, Bx] = arr.transpose()
  //caution: infinite
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def transpose(axes: Array[Int], dummy: Option[Boolean]): TFTensor[DType, Bx] = arr.transpose(Shape(axes: _*))
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def round(): TFTensor[DType, Ax]  = arr.round
  //Top-level slice only
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def slice(start: Int, end: Int, dummy: Option[Boolean]): TFTensor[DType, Bx] = arr(start :: end)

  //Caution: infinite

  //TODO: add this squeeze to base API
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def squeeze: TFTensor[DType, Bx] = arr.squeeze(Array(0)) 
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: TFTensor[DType, Ax]) def squeeze(index: Array[Int], dummy: Option[Boolean]): TFTensor[DType, Bx] =  arr.squeeze(index)

  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def rank: Int = arr.rank
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: TFTensor[DType], min: DType, max: DType): TFTensor[DType] = arr.clip(min, max) 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def unary_- : TFTensor[DType, Ax] = - arr 

//  def concat[DType <: Supported : ClassTag : IsSupported](axis: Int, arr: Seq[TFTensor[DType]]): TFTensor[DType] = ???
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[TFTensor[DType]]): TFTensor[DType] = ???

  //Also, time to get rid of classtag"?
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def abs(): TFTensor[DType, Ax] = arr.abs
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def ceil(): TFTensor[DType, Ax] = arr.ceil
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def floor(): TFTensor[DType, Ax] = arr.floor 

  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def log(): TFTensor[DType, Ax]= arr.log 
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def exp(): TFTensor[DType, Ax] = arr.exp
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def sqrt(): TFTensor[DType, Ax] = arr.sqrt
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def cos(): TFTensor[DType, Ax] = arr.cos
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def cosh(): TFTensor[DType, Ax] = arr.cosh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def sin(): TFTensor[DType, Ax] = arr.sin
  extension[DType <:FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def sinh(): TFTensor[DType, Ax] = arr.sinh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def tan(): TFTensor[DType, Ax] = arr.tan
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def tanh(): TFTensor[DType, Ax] = arr.tanh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def acos(): TFTensor[DType, Ax] = arr.acos
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def acosh(): TFTensor[DType, Ax] = arr.acosh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def asin(): TFTensor[DType, Ax] = arr.asin
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def asinh(): TFTensor[DType, Ax] = arr.asinh
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def atan(): TFTensor[DType, Ax] = arr.atan
  extension[DType <: FloatSupported: ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def atanh(): TFTensor[DType, Ax] = arr.atanh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: TFTensor[DType, Ax]) def sigmoid(): TFTensor[DType, Ax] = ???
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: TFTensor[DType, Ax]) def relu(): TFTensor[DType, Ax] = ???

  //Binary TFTensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def + (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr + other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def - (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr - other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def * (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr * other 
  extension[DType <: NumericSupported :  ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def ** (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr ** other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def / (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr / other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def % (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr % other 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def > (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr > other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def >= (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr >= other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def < (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr < other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def <= (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr <= other 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def ==== (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr === other 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def !=== (other: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[Boolean, Ax] = arr =!= other 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def max (d: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr maximum d 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes](arr: TFTensor[DType, Ax]) def min (d: TFTensor[DType, Ax])(implicit ev: Ax =!= Axes): TFTensor[DType, Ax] = arr minimum d 

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes, Bx <: Axes, Cx <: Axes](arr: TFTensor[DType, Ax]) def matmul (other: TFTensor[DType, Bx]): TFTensor[DType, Cx] = arr tensorDot (other, arr.rank)
}
