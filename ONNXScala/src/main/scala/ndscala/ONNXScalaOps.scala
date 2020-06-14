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
  implicit def convert[DType: ClassTag: Numeric](d: DType): NDArray[DType] = NDArray(ArraySeq(d), ArraySeq(1))
}

class ONNXScalaOps extends NDArrayOps[NDArray]{ 
  val onnx = new ORTOperatorBackendAll()
//  val seed = Array(1l,2l,3l,4l,5l)
  val rng = spire.random.rng.Cmwc5()

  //Nullary / factory ops
  def zeros[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): NDArray[DType] = NDArray(ArraySeq.fill(shape.product)(implicitly[Numeric[DType]].zero), shape)
  def ones[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): NDArray[DType] = NDArray(ArraySeq.fill(shape.product)(implicitly[Numeric[DType]].one), shape)
  def full[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int], value: DType): NDArray[DType] = NDArray(ArraySeq.fill(shape.product)(value), shape)
 
  //TODO: fix rand - generates Ints then widens
  def rand[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): NDArray[DType] = NDArray(ArraySeq.fill(shape.product)(rng.next[Int]), shape)

  //Unary ops
  def reshape[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], newShape: ArraySeq[Int]): NDArray[DType] = tensorToNDArray(onnx.Reshape5("reshape", Some(ndArrayToTensor(arr)), 
    Some(ndArrayToTensor(NDArray(newShape.map(x => x.toLong), ArraySeq(newShape.size))))))
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Transpose1("transpose", None, Some(ndArrayToTensor(arr))))
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], axes: ArraySeq[Int]): NDArray[DType] = tensorToNDArray(onnx.Transpose1("transpose", Some(axes.toArray), Some(ndArrayToTensor(arr)))) 
  def round[DType : ClassTag: Numeric: IsFloatSupported](arr: NDArray[DType]): NDArray[DType]  = tensorToNDArray(onnx.Round11("round", Some(ndArrayToTensor(arr))))
  //Top-level slice only
  def slice[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], start: Int, end: Int): NDArray[DType] = tensorToNDArray(onnx.Slice11("slice", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(NDArray(ArraySeq(start), ArraySeq(1)))), Some(ndArrayToTensor(NDArray(ArraySeq(end), ArraySeq(1))))))

  def squeeze[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], index: ArraySeq[Int]): NDArray[DType] = tensorToNDArray(onnx.Squeeze11("squeeze", Some(index.toArray), Some(ndArrayToTensor(arr))))
  def rank[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType]): Int = arr.shape.size
  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: NDArray[DType], min: DType, max: DType): NDArray[DType] = tensorToNDArray(onnx.Clip11("clip", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(NDArray(ArraySeq(min), ArraySeq[Int]()))), Some(ndArrayToTensor(NDArray(ArraySeq(max), ArraySeq[Int]())))))

  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType]) : NDArray[DType] = tensorToNDArray(onnx.Sub7("sub", Some(ndArrayToTensor(zeros(arr.shape))), Some(ndArrayToTensor(arr))))
  
  //Binary NDArray ops

  def +[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Add7("add", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def -[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Sub7("sub", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def *[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Mul7("mul", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def **[DType: ClassTag: Numeric: IsFloatSupported](arr: NDArray[DType], d: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Pow7("pow", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(d))))
  def /[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Div7("div", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def %[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Mod10("mod", None, Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))

  def >[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.Greater9[DType, Boolean]("gt", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def >=[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.GreaterOrEqual12[DType, Boolean]("gte", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def <[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.Less9[DType, Boolean]("lt", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def <=[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.LessOrEqual12[DType, Boolean]("lte", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def ===[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.Equal11[DType, Boolean]("eq", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))
  def !==[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[Boolean] = tensorToNDArray(onnx.Not1("not", Some(ndArrayToTensor(tensorToNDArray(onnx.Equal11[DType, Boolean]("eq", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other))))))))
 
  def max[DType: ClassTag: Numeric: IsFloatSupported](arr: NDArray[DType], d: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Max8("max", Seq(Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(d)))))
  def min[DType: ClassTag: Numeric: IsFloatSupported](arr: NDArray[DType], d: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.Min8("min", Seq(Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(d)))))

  def dot[DType : ClassTag: Numeric: IsSupported](arr: NDArray[DType], other: NDArray[DType]): NDArray[DType] = tensorToNDArray(onnx.MatMul9("matmul", Some(ndArrayToTensor(arr)), Some(ndArrayToTensor(other)))) 

  //TODO: implicit conversions
  private def ndArrayToTensor[DType: ClassTag](arr: NDArray[DType]): Tensor[DType] = {
    TensorFactory.getTensor(arr.data.toArray, arr.shape.toArray)
  }


  private def tensorToNDArray[DType: ClassTag](arr: Tensor[DType]): NDArray[DType] = {
    NDArray(scala.collection.immutable.ArraySeq[DType]() ++ arr._1, scala.collection.immutable.ArraySeq[Int]() ++ arr._2)
  }

}
