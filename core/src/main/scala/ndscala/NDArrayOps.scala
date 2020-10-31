package org.sciscala.ndscala

import org.sciscala.ndscala.union._
import scala.reflect.ClassTag
import spire.math.Numeric

//TODO: idea for named tensor / axis types : use string singleton types
//TODO: in simple-df : evaluate crossbow
//TODO: add buffer+slice lib
//Backend candidates: TF Java, scalapy-numpy/tf, ...
//case class NDArray[DType](data: Array[DType], shape: Array[Int])

//Should only allow supported here
trait NDArrayOps[SomeNDArray[_ <: AllSupported]] {
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
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): SomeNDArray[DType] 
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): SomeNDArray[DType] 
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int], value: DType): SomeNDArray[DType]
  //TOFIX
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): SomeNDArray[DType]

  //Unary ops
  //def reshape[DType <: Supported : ClassTag: Numeric](arr: SomeNDArray[DType], newShape: Array[Int]): SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported] (arr: SomeNDArray[DType]) def reShape(newShape: Array[Int]): SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported](arr: SomeNDArray[DType]) def transpose: SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported](arr: SomeNDArray[DType]) def transpose(axes: Array[Int], dummy: Option[Boolean]): SomeNDArray[DType] 
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType])  def round(): SomeNDArray[DType]
  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  extension[DType <: Supported : ClassTag : IsSupported] (arr: SomeNDArray[DType]) def slice(start: Int, end: Int, dummy: Option[Boolean]): SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported] (arr: SomeNDArray[DType]) def squeeze(index: Array[Int], dummy: Option[Boolean]): SomeNDArray[DType]
  extension[DType <: Supported : ClassTag : IsSupported] (arr: SomeNDArray[DType]) def rank: Int
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def unary_- : SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def abs(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def ceil(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def floor(): SomeNDArray[DType]
//  extension[DType <: Supported : ClassTag : IsSupported](arr: Seq[SomeNDArray[DType]]) def concat (axis: Int): SomeNDArray[DType]
 //TODO: reduceMean
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[SomeNDArray[DType]]): SomeNDArray[DType] 
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def log(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def exp(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def sqrt(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def cos(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def sin(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def tan(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def tanh(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def acos(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def asin(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def atan(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def atanh(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def sigmoid(): SomeNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: SomeNDArray[DType]) def relu(): SomeNDArray[DType]

  //Binary NDArray ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def matmul(other: SomeNDArray[DType]): SomeNDArray[DType]

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported] (arr: SomeNDArray[DType]) def +(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def -(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def *(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def **(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def /(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def %(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def >(other: SomeNDArray[DType]): SomeNDArray[Boolean]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def >=(other: SomeNDArray[DType]): SomeNDArray[Boolean]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def <(other: SomeNDArray[DType]): SomeNDArray[Boolean]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def <=(other: SomeNDArray[DType]): SomeNDArray[Boolean]

  //Restricted to numeric only because of TF
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def ====(other: SomeNDArray[DType]): SomeNDArray[Boolean]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def !===(other: SomeNDArray[DType]): SomeNDArray[Boolean]

  //TF-scala conflicts with max and min
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def max(other: SomeNDArray[DType]): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: SomeNDArray[DType]) def min(other: SomeNDArray[DType]): SomeNDArray[DType]

}
