package org.sciscala.ndscala

import org.sciscala.ndscala.union._
import scala.reflect.ClassTag
import spire.math.Numeric

import org.emergentorder.onnx.Tensors._
//import org.emergentorder.=!=
import io.kjaer.compiletime._
import io.kjaer.compiletime.Shape.NumElements
import org.emergentorder.compiletime._
import org.emergentorder.compiletime.TensorShapeDenotation.Reverse

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
  def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int], value: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  //TOFIX
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): SomeNDArray[DType]

  //Unary ops
  //def reshape[DType <: Supported : ClassTag: Numeric](arr: SomeNDArray[DType], newShape: Array[Int]): SomeNDArray[DType]
    //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: SomeNDArray[DType], newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType] 
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def reShape(newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): SomeNDArray[DType, (Tt1,Td1,S1)]

    extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): SomeNDArray[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])]
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def transpose(axes: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): SomeNDArray[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])]

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def slice(start: Int, end: Int, dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): SomeNDArray[DType, (Tt1,Td1,S1)]

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def squeeze(index: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): SomeNDArray[DType, (Tt1,Td1,S1)]
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def rank: Int


//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
 extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

//  extension[DType <: Supported : ClassTag : IsSupported](arr: Seq[SomeNDArray[DType]]) def concat (axis: Int): SomeNDArray[DType]
 //TODO: reduceMean
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[SomeNDArray[DType]]): SomeNDArray[DType] 
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def log()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def exp()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def sqrt()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def cos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def cosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def sin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def sinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def tan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def tanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def acos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def acosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def asin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def asinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def atan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def atanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def sigmoid()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def relu()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

  
//Binary NDArray ops
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def +(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def -(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def *(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def **(d: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def /(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def %(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def >(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def >=(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def <(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def <=(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]
  //Restricted to numeric only because of TF
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def ====(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def !===(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[Boolean, (Tt,Td,S)]

  //TF-scala conflicts with max and min
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def max(d: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def min(d: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil] (arr: SomeNDArray[DType, (Tt,Td,S)]) def matmul(other: SomeNDArray[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): SomeNDArray[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)]
}
