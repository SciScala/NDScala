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

  implicit def convert[DType <: Supported : ClassTag](d: DType): Tensor[DType, Vec[1, VecShape[1]]] = Tensor(Array(d), 1)
  implicit def toTensor[DType <: Supported : ClassTag](t: (Array[DType], Array[Int])): Tensor[DType, Axes] = Tensor.create(t._1.toArray, t._2.toArray)
  implicit def fromTensor[DType <: Supported : ClassTag, Ax <: Axes](t: Tensor[DType, Ax]): (Array[DType], Array[Int]) = {
    (t._1, t._2)
   
  }
}

//TODO: Stricter type bounds because ORT doesn't implement them all
//TODO: dotty-style typeclass
//Trying to load ORT shared lib from elsewhere
given NDArrayOps[OSTensor]{
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
  def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): Tensor[DType, Axes] = Tensor.create(Array.fill(shape.product)(implicitly[Numeric[DType]].zero), shape)
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int]): Tensor[DType, Axes] = Tensor.create(Array.fill(shape.product)(implicitly[Numeric[DType]].one), shape)
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported](shape: Array[Int], value: DType): Tensor[DType, Axes] = Tensor.create(Array.fill(shape.product)(value), shape)
 
  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): Tensor[DType] = ???

  //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: Tensor[DType], newShape: Array[Int]): Tensor[DType] 
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: Tensor[DType, Ax]) def reShape(newShape: Array[Int]): Tensor[DType, Bx] = onnx.ReshapeV5("reshape", arr,
    Tensor.create(newShape.toArray.map(x => x.toLong), Array(newShape.size)))

  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: Tensor[DType, Ax]) def transpose: Tensor[DType, Bx] = onnx.TransposeV1("transpose", None, arr)
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: Tensor[DType, Ax]) def transpose(axes: Array[Int], dummy: Option[Boolean]): Tensor[DType, Bx] = onnx.TransposeV1("transpose", Some(axes.toArray), arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def round(): Tensor[DType, Ax]  = onnx.RoundV11("round", arr)
  //Top-level slice only
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: Tensor[DType, Ax]) def slice(start: Int, end: Int, dummy: Option[Boolean]): Tensor[DType, Bx] = onnx.SliceV11("slice", arr, Tensor(Array(start), 1), Tensor(Array(end),1))

  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes, Bx <: Axes] (arr: Tensor[DType, Ax]) def squeeze(index: Array[Int], dummy: Option[Boolean]): Tensor[DType, Bx] = onnx.SqueezeV11("squeeze",Some(index.toArray),arr)
  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def rank: Int = arr._2.size

  //extension[DType <: FloatSupported : ClassTag: Numeric] (arr: Tensor[DType]) def clip(min: DType, max: DType): Tensor[DType] = onnx.ClipV11("clip", arr,None, None)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def unary_- : Tensor[DType, Ax] = onnx.NegV6("neg", arr)

  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def abs(): Tensor[DType, Ax] = onnx.AbsV6("abs", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def ceil(): Tensor[DType, Ax] = onnx.CeilV6("ceil", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def floor(): Tensor[DType, Ax] = onnx.FloorV6("floor", arr)
//  extension[DType <: Supported : ClassTag : IsSupported, Ax <: Axes](arr: Seq[Tensor[DType]]) def concat(axis: Int): Tensor[DType] = onnx.ConcatV11("concat", axis, arr)
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes](arr: Seq[Tensor[DType]]): Tensor[DType] = onnx.MeanV8("mean", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def log(): Tensor[DType, Ax]= onnx.LogV6("log", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def exp(): Tensor[DType, Ax] = onnx.ExpV6("exp", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def sqrt(): Tensor[DType, Ax] = onnx.SqrtV6("sqrt", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def cos(): Tensor[DType, Ax] = onnx.CosV7("cos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def cosh(): Tensor[DType, Ax] = onnx.CoshV9("cosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def sin(): Tensor[DType, Ax] = onnx.SinV7("sin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def sinh(): Tensor[DType, Ax] = onnx.SinhV9("sinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def tan(): Tensor[DType, Ax] = onnx.TanV7("tan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def tanh(): Tensor[DType, Ax] = onnx.TanhV6("tanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def acos(): Tensor[DType, Ax] = onnx.AcosV7("acos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def acosh(): Tensor[DType, Ax] = onnx.AcoshV9("acosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def asin(): Tensor[DType, Ax] = onnx.AsinV7("asin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def asinh(): Tensor[DType, Ax] = onnx.AsinhV9("asinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def atan(): Tensor[DType, Ax] = onnx.AtanV7("atan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def atanh(): Tensor[DType, Ax] = onnx.AtanhV9("atanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def sigmoid(): Tensor[DType, Ax] = onnx.SigmoidV6("sig", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def relu(): Tensor[DType, Ax] = onnx.ReluV6("relu", arr)


  //Binary Tensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def +(other: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.AddV7("add", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def -(other: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.SubV7("sub", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def *(other: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.MulV7("mul", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def **(d: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.PowV7("pow", arr, d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def /(other: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.DivV7("div", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def %(other: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.ModV10("mod", None, arr, other)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def >(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.GreaterV9[DType, Boolean, Ax]("gt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def >=(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.GreaterOrEqualV12[DType, Boolean, Ax]("gte", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def <(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.LessV9[DType, Boolean, Ax]("lt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def <=(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.LessOrEqualV12[DType, Boolean, Ax]("lte", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def ====(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.EqualV11[DType, Boolean, Ax]("eq", arr, other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def !===(other: Tensor[DType, Ax]): Tensor[Boolean, Ax] = onnx.NotV1("not", onnx.EqualV11[DType, Boolean, Ax]("eq", arr, other))
 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def max(d: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.MaxV8("max", Seq(arr, d))
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes] (arr: Tensor[DType, Ax]) def min(d: Tensor[DType, Ax]): Tensor[DType, Ax] = onnx.MinV8("min", Seq(arr, d))

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Ax <: Axes, Bx <: Axes, Cx <: Axes] (arr: Tensor[DType, Ax]) def matmul(other: Tensor[DType, Bx]): Tensor[DType, Cx] = onnx.MatMulV9("matmul", arr, other)
}
