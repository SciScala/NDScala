package org.sciscala.ndscala 

import scala.util.Random
//import scala.collection.immutable.Array
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math._
//import spire.implicits.Numeric
import scala.language.implicitConversions

import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends.ORTOperatorBackendAll

object ONNXScalaOps {

  implicit def convert[DType <: Supported : ClassTag](d: DType): Tensor[DType] = (Array(d).toArray, Array(1).toArray)
  implicit def toTensor[DType <: Supported : ClassTag](t: (Array[DType], Array[Int])): Tensor[DType] = (t._1.toArray, t._2.toArray)
  implicit def fromTensor[DType <: Supported : ClassTag](t: Tensor[DType]): (Array[DType], Array[Int]) = {
    val onnxTensor = getOnnxTensor(t._1, t._2)
    (getArrayFromOnnxTensor(onnxTensor), onnxTensor.getInfo.getShape.map(_.toInt))
  }
}

//TODO: Stricter type bounds because ORT doesn't implement them all
//TODO: dotty-style typeclass
//Trying to load ORT shared lib from elsewhere
given NDArrayOps[Tensor]{
//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  implicit def convert[DType <: Supported : ClassTag](d: DType): Tensor[DType] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
//  implicit def toTensor[DType <: Supported : ClassTag](t: (Array[DType], Array[Int])): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromTensor[DType <: Supported : ClassTag](t: Tensor[DType]): (Array[DType], Array[Int]) = (t._1, t._2)
/*
  given convert as Conversion[Int, Tensor[Int]] {
    def apply(d:Int): Tensor[Int] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertLong as Conversion[Long, Tensor[Long]] {
    def apply(d:Long): Tensor[Long] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertFloat as Conversion[Float, Tensor[Float]] {
    def apply(d:Float): Tensor[Float] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertDouble as Conversion[Double, Tensor[Double]] {
    def apply(d:Double): Tensor[Double] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given toTensor as Conversion[(Array[Int], Array[Int]), Tensor[Int]]{
    def apply(x: (Array[Int], Array[Int])): Tensor[Int] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorLong as Conversion[(Array[Long], Array[Int]), Tensor[Long]]{
    def apply(x: (Array[Long], Array[Int])): Tensor[Long] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }
  given toTensorFloat as Conversion[(Array[Float], Array[Int]), Tensor[Float]]{
    def apply(x: (Array[Float], Array[Int])): Tensor[Float] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorDouble as Conversion[(Array[Double], Array[Int]), Tensor[Double]]{
    def apply(x: (Array[Double], Array[Int])): Tensor[Double] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

 
  given fromTensor as Conversion[Tensor[Int], (Array[Int], Array[Int])]{
    def apply(x: Tensor[Int]):(Array[Int], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorLong as Conversion[Tensor[Long], (Array[Long], Array[Int])]{
    def apply(x: Tensor[Long]):(Array[Long], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorFloat as Conversion[Tensor[Float], (Array[Float], Array[Int])]{
    def apply(x: Tensor[Float]):(Array[Float], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorDouble as Conversion[Tensor[Double], (Array[Double], Array[Int])]{
    def apply(x: Tensor[Double]):(Array[Double], Array[Int]) = (Array.from(x._1), Array.from(x._2))
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
  def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): Tensor[DType] = (Array.fill(shape.product)(implicitly[Numeric[DType]].zero), shape)
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): Tensor[DType] = (Array.fill(shape.product)(implicitly[Numeric[DType]].one), shape)
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int], value: DType): Tensor[DType] = (Array.fill(shape.product)(value), shape)
 
  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): Tensor[DType] = ???

  //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: Tensor[DType], newShape: Array[Int]): Tensor[DType] 
  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def reShape(newShape: Array[Int]): Tensor[DType] = onnx.ReshapeV5("reshape", arr,
    (newShape.toArray.map(x => x.toLong), Array(newShape.size)))

  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def transpose: Tensor[DType] = onnx.TransposeV1("transpose", None, arr)
  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def transpose(axes: Array[Int], dummy: Option[Boolean]): Tensor[DType] = onnx.TransposeV1("transpose", Some(axes.toArray), arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def round(): Tensor[DType]  = onnx.RoundV11("round", arr)
  //Top-level slice only
  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def slice(start: Int, end: Int, dummy: Option[Boolean]): Tensor[DType] = onnx.SliceV11[DType, Int]("slice", arr, (Array(start), Array(1)), (Array(end), Array(1)))

  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def squeeze(index: Array[Int], dummy: Option[Boolean]): Tensor[DType] = onnx.SqueezeV11("squeeze",Some(index.toArray),arr)
  extension[DType <: Supported : ClassTag : IsSupported] (arr: Tensor[DType]) def rank: Int = getOnnxTensor(arr._1, arr._2).getInfo.getShape.size

  //extension[DType <: FloatSupported : ClassTag: Numeric] (arr: Tensor[DType]) def clip(min: DType, max: DType): Tensor[DType] = onnx.ClipV11("clip", arr,None, None)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def unary_- : Tensor[DType] = onnx.SubV7("sub", zeros(getOnnxTensor(arr._1, arr._2).getInfo.getShape.map(_.toInt)), arr)

  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: Tensor[DType]) def abs(): Tensor[DType] = onnx.AbsV6("abs", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def ceil(): Tensor[DType] = onnx.CeilV6("ceil", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def floor(): Tensor[DType] = onnx.FloorV6("floor", arr)
//  extension[DType <: Supported : ClassTag : IsSupported](arr: Seq[Tensor[DType]]) def concat(axis: Int): Tensor[DType] = onnx.ConcatV11("concat", axis, arr)
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[Tensor[DType]]): Tensor[DType] = onnx.MeanV8("mean", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def log(): Tensor[DType]= onnx.LogV6("log", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def exp(): Tensor[DType] = onnx.ExpV6("exp", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def sqrt(): Tensor[DType] = onnx.SqrtV6("sqrt", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def cos(): Tensor[DType] = onnx.CosV7("cos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def cosh(): Tensor[DType] = onnx.CoshV9("cosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def sin(): Tensor[DType] = onnx.SinV7("sin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def sinh(): Tensor[DType] = onnx.SinhV9("sinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def tan(): Tensor[DType] = onnx.TanV7("tan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def tanh(): Tensor[DType] = onnx.TanhV6("tanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def acos(): Tensor[DType] = onnx.AcosV7("acos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def acosh(): Tensor[DType] = onnx.AcoshV9("acosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def asin(): Tensor[DType] = onnx.AsinV7("asin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def asinh(): Tensor[DType] = onnx.AsinhV9("asinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def atan(): Tensor[DType] = onnx.AtanV7("atan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def atanh(): Tensor[DType] = onnx.AtanhV9("atanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def sigmoid(): Tensor[DType] = onnx.SigmoidV6("sig", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: Tensor[DType]) def relu(): Tensor[DType] = onnx.ReluV6("relu", arr)


  //Binary Tensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def +(other: Tensor[DType]): Tensor[DType] = onnx.AddV7("add", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def -(other: Tensor[DType]): Tensor[DType] = onnx.SubV7("sub", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def *(other: Tensor[DType]): Tensor[DType] = onnx.MulV7("mul", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def **(d: Tensor[DType]): Tensor[DType] = onnx.PowV7("pow", arr, d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def /(other: Tensor[DType]): Tensor[DType] = onnx.DivV7("div", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def %(other: Tensor[DType]): Tensor[DType] = onnx.ModV10("mod", None, arr, other)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def >(other: Tensor[DType]): Tensor[Boolean] = onnx.GreaterV9[DType, Boolean]("gt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def >=(other: Tensor[DType]): Tensor[Boolean] = onnx.GreaterOrEqualV12[DType, Boolean]("gte", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def <(other: Tensor[DType]): Tensor[Boolean] = onnx.LessV9[DType, Boolean]("lt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def <=(other: Tensor[DType]): Tensor[Boolean] = onnx.LessOrEqualV12[DType, Boolean]("lte", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: Tensor[DType]) def ====(other: Tensor[DType]): Tensor[Boolean] = onnx.EqualV11[DType, Boolean]("eq", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: Tensor[DType]) def !===(other: Tensor[DType]): Tensor[Boolean] = onnx.NotV1("not", onnx.EqualV11[DType, Boolean]("eq", arr, other))
 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def max(d: Tensor[DType]): Tensor[DType] = onnx.MaxV8("max", Seq(arr, d))
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def min(d: Tensor[DType]): Tensor[DType] = onnx.MinV8("min", Seq(arr, d))

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported] (arr: Tensor[DType]) def matmul(other: Tensor[DType]): Tensor[DType] = onnx.MatMulV9("matmul", arr, other)
}
