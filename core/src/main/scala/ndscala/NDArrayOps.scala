package org.sciscala.ndscala

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import org.sciscala.ndscala.union._
import spire.math._

import simulacrum._

//Backend candidates: TF Java, ND4J(OpenBlas), MXNet, scalapy-numpy/tf, ...
case class NDArray[DType](data: ArraySeq[DType], shape: ArraySeq[Int])

@typeclass trait NDArrayOps[SomeNDArray[_]] {
  type Supported = Union[Int]#or[Long]#or[Float]#or[Double]#create
  type FloatSupported = Union[Float]#or[Double]#create
  type IsSupported[T] = Contains[T, Supported]
  type IsFloatSupported[T] = Contains[T, FloatSupported]

  //TODO: missing ops
  // numpy does: arange/range, exp, sqrt, all trig functions, log, 
  // concat, floor, ceil, square, abs, argmax/min, pad, ...
  
  //Nullary / factory ops
  def zeros[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): SomeNDArray[DType] 
  def ones[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): SomeNDArray[DType] 
  def full[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int], value: DType): SomeNDArray[DType]
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
  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: SomeNDArray[DType]) : SomeNDArray[DType]

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
