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
import org.emergentorder.compiletime.TensorShapeDenotation.Concat

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
  extension [DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](arr: SomeNDArray[DType, (Tt,Td,S)]) def reshape[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): SomeNDArray[DType, (Tt1,Td1,S1)]

    extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): SomeNDArray[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])]
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def transpose(axes: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): SomeNDArray[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])]

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

  //TODO: add unsqueeze, broaden slice, extra sugar for slice, range, squeeze, ...
  //Squeeze could take axes as a type param, and derive the output shape
//  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def slice[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](start: Int, end: Int)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): SomeNDArray[DType, (Tt1,Td1,S1)]

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: SomeNDArray[DType, (Tt,Td,S)]) def slice[Tt2 <: TensorTypeDenotation, AxesStart <: Indices, AxesEnd <: Indices](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td], s2: ShapeOf[SlicedShape[AxesStart,AxesEnd]], i: IndicesOf[AxesStart], i2: IndicesOf[AxesEnd]): SomeNDArray[DType, (Tt2,Td,SlicedShape[AxesStart,AxesEnd])]

  //TODO: use Reduce type to remove squeezed axes
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def squeeze[Tt1 <: TensorTypeDenotation, Axes <: Indices](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,false]], s: ShapeOf[KeepOrReduceDims[S,Axes,false]], i: IndicesOf[Axes]): SomeNDArray[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,false],KeepOrReduceDims[S,Axes,false]]]
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def rank: Int


//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: SomeNDArray[DType], min: DType, max: DType): SomeNDArray[DType]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, SSuffix <: Shape, S <: Dimension #: SSuffix, S1 <: Dimension #: SSuffix](arr: Tuple2[SomeNDArray[DType, (Tt, Td, S)],SomeNDArray[DType, (Tt, Td, S1)]]) def concat[Axis <: Index ::: INil](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[AddGivenAxisSize[S,S1,Axis]], i: IndicesOf[Axis]): SomeNDArray[DType, (Tt,Td,AddGivenAxisSize[S,S1,Axis])]


  //Reduction ops:
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def reduceSum[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): SomeNDArray[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]]

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Seq[SomeNDArray[DType, (Tt,Td,S)]]) def mean()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)] 
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

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def inverse()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def constant()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  
//Binary NDArray ops
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def +(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def -(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def *(other: SomeNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)]
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, DType1 <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: SomeNDArray[DType, (Tt,Td,S)]) def **(d: SomeNDArray[DType1, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): SomeNDArray[DType, (Tt,Td,S)] 
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
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil] (arr: SomeNDArray[DType, (Tt,Td,S)]) def matmul[Dim2 <: Dimension, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil](other: SomeNDArray[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td],vd:ValueOf[scala.compiletime.ops.int.S[Dim0]], vd1:ValueOf[scala.compiletime.ops.int.S[Dim1]], vd2: ValueOf[scala.compiletime.ops.int.S[Dim2]], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): SomeNDArray[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)]
}
