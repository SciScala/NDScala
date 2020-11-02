package org.sciscala.ndscala

import org.sciscala.ndscala.union._
import scala.reflect.ClassTag
import spire.math.Numeric
import org.emergentorder.onnx.Tensors._

//TODO: idea for named tensor / axis types : use string singleton types
//TODO: in simple-df : evaluate crossbow
//TODO: add buffer+slice lib
//Backend candidates: TF Java, scalapy-numpy/tf, ...
//case class NDArray[DType](data: Array[DType], shape: Array[Int])

//Should only allow supported here
trait NDArrayOps[SomeNDArray[_ <: AllSupported, _ <: Axes]] {
  //TODO: factor out new-style unions to a dotty-specific file
  type NumericSupported = Int | Long | Float | Double
  type Supported = AllSupported 
  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

  
  type UnionNumericSupported = Union[Int]#or[Long]#or[Float]#or[Double]#create 
  type UnionSupported = Union[Int]#or[Long]#or[Float]#or[Double]#or[Boolean]#create
  type UnionFloatSupported = Union[Float]#or[Double]#create
  type IsNumericSupported[T] = Contains[T, UnionNumericSupported]
  type IsSupported[T] = Contains[T, UnionSupported]
  type IsFloatSupported[T] = Contains[T, UnionFloatSupported]

  //TODO: missing ops
  // numpy does: pad, range, concat, square(Missing), argmax/min
 
  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): SomeNDArray[DType, Axes] 
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): SomeNDArray[DType, Axes] 
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int], value: DType): SomeNDArray[DType, Axes]
  //TOFIX
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): SomeNDArray[DType]

  //Unary ops
  //def reshape[DType <: Supported : ClassTag: Numeric](arr: SomeNDArray[DType], newShape: Array[Int]): SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: SomeNDArray[DType, Ax]) def reShape(newShape: Array[Int]): SomeNDArray[DType, Bx]
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: SomeNDArray[DType, Ax]) def transpose: SomeNDArray[DType, Bx]
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes](arr: SomeNDArray[DType, Ax]) def transpose(axes: Array[Int], dummy: Option[Boolean]): SomeNDArray[DType,Bx]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax])  def round(): SomeNDArray[DType, Ax]
  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: SomeNDArray[DType, Ax]) def slice(start: Int, end: Int, dummy: Option[Boolean]): SomeNDArray[DType, Bx]
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: SomeNDArray[DType, Ax]) def squeeze(index: Array[Int], dummy: Option[Boolean]): SomeNDArray[DType, Bx]
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def rank: Int
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def unary_- : SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def abs(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType,Ax]) def ceil(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def floor(): SomeNDArray[DType, Ax]
//  extension[DType <: Supported : ClassTag : IsSupported](arr: Seq[SomeNDArray[DType]]) def concat (axis: Int): SomeNDArray[DType]
 //TODO: reduceMean
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[SomeNDArray[DType]]): SomeNDArray[DType] 
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def log(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def exp(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def sqrt(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def cos(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def sin(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def tan(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def tanh(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def acos(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def asin(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def atan(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def atanh(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def sigmoid(): SomeNDArray[DType, Ax]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def relu(): SomeNDArray[DType, Ax]

  //Binary NDArray ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes, Bx <: Axes, Cx <: Axes] (arr: SomeNDArray[DType, Ax]) def matmul(other: SomeNDArray[DType, Bx]): SomeNDArray[DType, Cx]

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def +(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def -(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def *(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def **(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def /(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def %(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def >(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def >=(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def <(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def <=(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]

  //Restricted to numeric only because of TF
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def ====(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def !===(other: SomeNDArray[DType, Ax]): SomeNDArray[Boolean, Ax]

  //TF-scala conflicts with max and min
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def max(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: SomeNDArray[DType, Ax]) def min(other: SomeNDArray[DType, Ax]): SomeNDArray[DType, Ax]

}
