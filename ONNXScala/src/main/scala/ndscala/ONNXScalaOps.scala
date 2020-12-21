package org.sciscala.ndscala 

import scala.util.Random
//import scala.collection.immutable.Array
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math._
//import spire.implicits.Numeric
import scala.language.implicitConversions

import io.kjaer.compiletime._
import io.kjaer.compiletime.Shape.NumElements
import org.emergentorder.compiletime._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends.ORTOperatorBackendAll
import org.emergentorder.compiletime.TensorShapeDenotation.Reverse
//import org.emergentorder.=!=

object ONNXScalaOps {

  //implicit def convert[DType <: Supported : ClassTag](d: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (?, ?, 1 #: SNil)] = Tensor(Array(d), (?,?,1 #: SNil))
//  implicit def toTensor[DType <: Supported : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: (Array[DType], (Tt,Td,S)))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(t.data.toIAy,)
//  implicit def fromTensor[DType <: Supported : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): (Array[DType], (Tt,Td,S)) = {
//    (t.data, t._2)
//   
//  }
}

//TODO: Stricter type bounds because ORT doesn't implement them all
//TODO: dotty-style typeclass
//Trying to load ORT shared lib from elsewhere
given NDArrayOps[Tensor] with {
//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  implicit def convert[DType <: Supported : ClassTag](d: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
//  implicit def toTensor[DType <: Supported : ClassTag](t: (Array[DType], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromTensor[DType <: Supported : ClassTag](t: Tensor[DType])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): (Array[DType], Array[Int]) = (t._1, t._2)
/*
  given convert as Conversion[Int, Tensor[Int]] {
    def apply(d:Int)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Int] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertLong as Conversion[Long, Tensor[Long]] {
    def apply(d:Long)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertFloat as Conversion[Float, Tensor[Float]] {
    def apply(d:Float)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Float] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertDouble as Conversion[Double, Tensor[Double]] {
    def apply(d:Double)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Double] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given toTensor as Conversion[(Array[Int], Array[Int]), Tensor[Int]]{
    def apply(x: (Array[Int], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Int] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorLong as Conversion[(Array[Long], Array[Int]), Tensor[Long]]{
    def apply(x: (Array[Long], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }
  given toTensorFloat as Conversion[(Array[Float], Array[Int]), Tensor[Float]]{
    def apply(x: (Array[Float], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Float] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorDouble as Conversion[(Array[Double], Array[Int]), Tensor[Double]]{
    def apply(x: (Array[Double], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Double] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

 
  given fromTensor as Conversion[Tensor[Int], (Array[Int], Array[Int])]{
    def apply(x: Tensor[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Int], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorLong as Conversion[Tensor[Long], (Array[Long], Array[Int])]{
    def apply(x: Tensor[Long])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Long], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorFloat as Conversion[Tensor[Float], (Array[Float], Array[Int])]{
    def apply(x: Tensor[Float])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Float], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorDouble as Conversion[Tensor[Double], (Array[Double], Array[Int])]{
    def apply(x: Tensor[Double])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Double], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }
*/


//class ONNXScalaOps extends NDArrayOps[Tensor]{
//  import ONNXScalaOps._
 /*
  import org.sciscala.ndscala.ONNXScalaOps.toTensor
  import org.sciscala.ndscala.ONNXScalaOps.toTensorLong 
  import org.sciscala.ndscala.ONNXScalaOps.toTensorFloat
  import org.sciscala.ndscala.ONNXScalaOps.toTensorDouble

  import org.sciscala.ndscala.ONNXScalaOps.fromTensor
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorLong 
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorFloat
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorDouble

  import org.sciscala.ndscala.ONNXScalaOps.convert
  import org.sciscala.ndscala.ONNXScalaOps.convertLong 
  import org.sciscala.ndscala.ONNXScalaOps.convertFloat
  import org.sciscala.ndscala.ONNXScalaOps.convertDouble
*/
  val onnx = new ORTOperatorBackendAll()
//  val seed = Array(1l,2l,3l,4l,5l)
  val rng = spire.random.rng.Cmwc5()

  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(s.value.toSeq.foldLeft(1)(_*_))(implicitly[Numeric[DType]].zero),tt.value,td.value, s.value)
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(shape.foldLeft(1)(_*_))(implicitly[Numeric[DType]].one), tt.value, td.value,s.value)
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int], value: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(shape.foldLeft(1)(_*_))(value), tt.value, td.value, s.value)
 
  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = ???

  //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: Tensor[DType], newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] 
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reShape(newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): Tensor[DType, (Tt1,Td1,S1)] = onnx.ReshapeV5("reshape", arr,
    Tensor(newShape.toArray.map(x => x.toLong), tt1.value, td1.value, Shape.fromSeq(Array(newShape.size)))) //wrong denotations

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): Tensor[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = onnx.TransposeV1("transpose", None, arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def transpose(axes: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): Tensor[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = onnx.TransposeV1("transpose", Some(axes.toArray), arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)]  = onnx.RoundV11("round", arr)
  //Top-level slice only
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def slice(start: Int, end: Int, dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): Tensor[DType, (Tt1,Td1,S1)] = onnx.SliceV11("slice", arr, Tensor(Array(start), tt1.value, td1.value, 1 #: SNil), Tensor(Array(end), tt1.value, td1.value,1 #: SNil)) //wrong denotations

  //TODO: Add to core, restrict starts and ends to vector
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape, Tt2 <: TensorTypeDenotation, Td2 <: TensorShapeDenotation, S2 <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def slice(start: Tensor[Int, (Tt1,Td1, S1)], end: Tensor[Int, (Tt1,Td1, S1)], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt2], td1: TensorShapeDenotationOf[Td2], s2: ShapeOf[S2]): Tensor[DType, (Tt2,Td2,S2)] = onnx.SliceV11("slice", arr, start, end) //wrong denotations

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def squeeze(index: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): Tensor[DType, (Tt1,Td1,S1)] = onnx.SqueezeV11("squeeze",Some(index.toArray),arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def rank: Int = arr.shape.size

  //extension[DType <: FloatSupported : ClassTag: Numeric] (arr: Tensor[DType]) def clip(min: DType, max: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = onnx.ClipV11("clip", arr,None, None)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.NegV6("neg", arr)

  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AbsV6("abs", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CeilV6("ceil", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.FloorV6("floor", arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](arr: Seq[Tensor[DType, (Tt,Td,S)]]) def concat(axis: Int)(using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[Td1], s: ShapeOf[S1]): Tensor[DType, (Tt1, Td1, S1)] = onnx.ConcatV11("concat", axis, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](arr: Seq[Tensor[DType, (Tt,Td,S)]]) def mean()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MeanV8("mean", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def log()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)]= onnx.LogV6("log", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def exp()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ExpV6("exp", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sqrt()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SqrtV6("sqrt", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def cos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CosV7("cos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def cosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CoshV9("cosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SinV7("sin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SinhV9("sinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def tan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.TanV7("tan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def tanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.TanhV6("tanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def acos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AcosV7("acos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def acosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AcoshV9("acosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def asin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AsinV7("asin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def asinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AsinhV9("asinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def atan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AtanV7("atan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def atanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AtanhV9("atanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sigmoid()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SigmoidV6("sig", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def relu()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ReluV6("relu", arr)


  //Binary Tensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def +(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AddV7("add", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def -(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SubV7("sub", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def *(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MulV7("mul", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def **(d: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.PowV7("pow", arr, d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def /(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.DivV7("div", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def %(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ModV10("mod", None, arr, other)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def >(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.GreaterV9[DType, Boolean, Tt,Td,S, Tt, Td]("gt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def >=(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.GreaterOrEqualV12[DType, Boolean, Tt,Td,S, Tt, Td]("gte", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def <(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.LessV9[DType, Boolean, Tt,Td,S, Tt, Td]("lt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def <=(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.LessOrEqualV12[DType, Boolean, Tt,Td,S, Tt, Td]("lte", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ====(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.EqualV11[DType, Boolean, Tt,Td,S, Tt, Td]("eq", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def !===(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.NotV1("not", onnx.EqualV11[DType, Boolean, Tt,Td,S, Tt, Td]("eq", arr, other))
 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def max(d: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MaxV8("max", Seq(arr, d))
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def min(d: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MinV8("min", Seq(arr, d))

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def matmul(other: Tensor[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): Tensor[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)] = onnx.MatMulV9[DType, Dim0, Dim1, Dim2, Tt,Td,S,Tt1,Td1,S1]("matmul", arr, other)
}
