package org.sciscala.ndscala

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import org.sciscala.ndscala.union._
import spire.math._

import simulacrum._

//Backend candidates: TF Java, ND4J(OpenBlas), MXNet, scalapy-numpy/tf, ...
//case class NDArray[DType](data: ArraySeq[DType], shape: ArraySeq[Int])

@typeclass trait NDArrayOps[SomeNDArray[_]] {
  type Supported = Union[Int]#or[Long]#or[Float]#or[Double]#create
  type FloatSupported = Union[Float]#or[Double]#create
  type IsSupported[T] = Contains[T, Supported]
  type IsFloatSupported[T] = Contains[T, FloatSupported]

  //TODO: missing ops
  // numpy does: pad, range, concat, square(Missing), argmax/min
 
  //Nullary / factory ops
//  def zeros[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): SomeNDArray[DType] 
//  def ones[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): SomeNDArray[DType] 
//  def full[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int], value: DType): SomeNDArray[DType]
  def rand[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): SomeNDArray[DType]

  //Unary ops
  def reshape[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], newShape: ArraySeq[Int]): SomeNDArray[DType]
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], axes: ArraySeq[Int]): SomeNDArray[DType] 
  def round[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  def slice[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], start: Int, end: Int): SomeNDArray[DType]
  def squeeze[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], index: ArraySeq[Int]): SomeNDArray[DType]
  def rank[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]): Int
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]) : SomeNDArray[DType]


  def abs[DType: ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def ceil[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def floor[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]*): SomeNDArray[DType]
  def log[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def exp[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def sqrt[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def cos[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def sin[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def tan[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def tanh[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def acos[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def asin[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
  def atan[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]
//  def atanh[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType]): SomeNDArray[DType]


  //Binary NDArray ops

  def dot[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]

  def +[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def -[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def *[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def **[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def /[DType: ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def %[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def >[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]
  def >=[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]
  def <[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]
  def <=[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]

  def ===[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]
  def !==[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[Boolean]

  def max[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]
  def min[DType: ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], other: SomeNDArray[DType]): SomeNDArray[DType]

}
