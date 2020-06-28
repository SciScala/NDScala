package org.sciscala.ndscala 

import scala.util.Random
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math._
import spire.implicits._

import org.emergentorder.onnx.{Tensor, TensorFactory}
import org.emergentorder.onnx.backends.ORTOperatorBackendAll

object ONNXScalaOps {
  implicit def convert[DType: ClassTag: Numeric](d: DType): Tensor[DType] = TensorFactory.getTensor(ArraySeq(d).toArray, ArraySeq(1).toArray)
  implicit def toTensor[DType: ClassTag: Numeric](t: (ArraySeq[DType], ArraySeq[Int])): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
  implicit def fromTensor[DType: ClassTag](t: Tensor[DType]): (ArraySeq[DType], ArraySeq[Int]) = (ArraySeq.from(t._1), ArraySeq.from(t._2))
}

class ONNXScalaOps extends NDArrayOps[Tensor]{
  import ONNXScalaOps._
  val onnx = new ORTOperatorBackendAll()
//  val seed = Array(1l,2l,3l,4l,5l)
  val rng = spire.random.rng.Cmwc5()

  //Nullary / factory ops
  def zeros[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): Tensor[DType] = (ArraySeq.fill(shape.product)(implicitly[Numeric[DType]].zero), shape)
  def ones[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): Tensor[DType] = (ArraySeq.fill(shape.product)(implicitly[Numeric[DType]].one), shape)
  def full[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int], value: DType): Tensor[DType] = (ArraySeq.fill(shape.product)(value), shape)
 
  //TODO: fix rand
  def rand[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): Tensor[DType] = ???

  //Unary ops
  def reshape[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], newShape: ArraySeq[Int]): Tensor[DType] = onnx.Reshape5("reshape", Some(arr),
    Some((newShape.map(x => x.toLong), ArraySeq(newShape.size))))
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Transpose1("transpose", None, Some(arr))
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], axes: ArraySeq[Int]): Tensor[DType] = onnx.Transpose1("transpose", Some(axes.toArray), Some(arr))
  def round[DType : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType]  = onnx.Round11("round", Some(arr))
  //Top-level slice only
  def slice[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], start: Int, end: Int): Tensor[DType] = onnx.Slice11[DType, Int]("slice", Some(arr), Some((ArraySeq(start), ArraySeq(1))), Some((ArraySeq(end), ArraySeq(1))))

  def squeeze[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], index: ArraySeq[Int]): Tensor[DType] = onnx.Squeeze11("squeeze", Some(index.toArray), Some(arr))
  def rank[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Int = arr._2.size
  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], min: DType, max: DType): Tensor[DType] = onnx.Clip11("clip", Some(arr), Some((ArraySeq(min), ArraySeq[Int]())), Some((ArraySeq(max), ArraySeq[Int]())))

  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType]) : Tensor[DType] = onnx.Sub7("sub", Some(zeros(ArraySeq.from(arr._2))), Some(arr))

  def abs[DType: ClassTag: Numeric: IsSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Abs6("abs", Some(arr))
  def ceil[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Ceil1("ceil", None, Some(arr))
  def floor[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Floor1("floor", None, Some(arr))
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: Tensor[DType]*): Tensor[DType] = onnx.Concat11("concat", Some(ndArrayToTensor(arr)))))
  def log[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType]= onnx.Log6("log", Some(arr))
  def exp[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Exp6("exp", Some(arr))
  def sqrt[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Sqrt1("sqrt", None, Some(arr))
  def cos[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Cos7("cos", Some(arr))
  def cosh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Cosh9("cosh", Some(arr))
  def sin[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Sin7("sin", Some(arr))
  def sinh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Sinh9("sinh", Some(arr))
  def tan[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Tan7("tan", Some(arr))
  def tanh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Tanh6("tanh", Some(arr))
  def acos[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Acos7("acos", Some(arr))
  def acosh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Acosh9("acosh", Some(arr))
  def asin[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Asin7("asin", Some(arr))
  def asinh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Asinh9("asinh", Some(arr))
  def atan[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Atan7("atan", Some(arr))
//  def atanh[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType]): Tensor[DType] = onnx.Atanh9("atanh", Some(arr))


  //Binary Tensor ops

  def +[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.Add7("add", Some(arr), Some(other))
  def -[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.Sub7("sub", Some(arr), Some(other))
  def *[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.Mul7("mul", Some(arr), Some(other))
  def **[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], d: Tensor[DType]): Tensor[DType] = onnx.Pow7("pow", Some(arr), Some(d))
  def /[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.Div7("div", Some(arr), Some(other))
  def %[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.Mod10("mod", None, Some(arr), Some(other))

  def >[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.Greater9[DType, Boolean]("gt", Some(arr), Some(other))
  def >=[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.GreaterOrEqual12[DType, Boolean]("gte", Some(arr), Some(other))
  def <[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.Less9[DType, Boolean]("lt", Some(arr), Some(other))
  def <=[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.LessOrEqual12[DType, Boolean]("lte", Some(arr), Some(other))
  def ===[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.Equal11[DType, Boolean]("eq", Some(arr), Some(other))
  def !==[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[Boolean] = onnx.Not1("not", Some(onnx.Equal11[DType, Boolean]("eq", Some(arr), Some(other))))
 
  def max[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], d: Tensor[DType]): Tensor[DType] = onnx.Max8("max", Seq(Some(arr), Some(d)))
  def min[DType: ClassTag: Numeric: IsFloatSupported](arr: Tensor[DType], d: Tensor[DType]): Tensor[DType] = onnx.Min8("min", Seq(Some(arr), Some(d)))

  def dot[DType : ClassTag: Numeric: IsSupported](arr: Tensor[DType], other: Tensor[DType]): Tensor[DType] = onnx.MatMul9("matmul", Some(arr), Some(other))
}
